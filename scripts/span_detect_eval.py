import json
from collections import defaultdict
import regex as re
import os

propaganda_techniques = ['Appeal_to_Values', 'Loaded_Language', 'Consequential_Oversimplification',
                         'Causal_Oversimplification', 'Questioning_the_Reputation', 'Straw_Man', 'Repetition',
                         'Guilt_by_Association', 'Appeal_to_Hypocrisy', 'Conversation_Killer',
                         'False_Dilemma-No_Choice', 'Whataboutism', 'Slogans',
                         'Obfuscation-Vagueness-Confusion',
                         'Name_Calling-Labeling', 'Flag_Waving', 'Doubt',
                         'Appeal_to_Fear-Prejudice', 'Exaggeration-Minimisation', 'Red_Herring',
                         'Appeal_to_Popularity', 'Appeal_to_Authority', 'Appeal_to_Time']


def load_json_as_list(fname, correctStart, isGold):
    labels_per_par = defaultdict(list)

    with open(fname, 'r', encoding="utf-8") as inf:
        for i,line in enumerate(inf):
            jobj = json.loads(line)
            par_id = jobj['paragraph_id'] if 'paragraph_id' in jobj else jobj['id']

            if isGold:
                labels = jobj['labels']
            else:
                labels = jobj['llm_annotations']

            per_par_labels = []
            if type(labels) != list:
                labels = []

            for label in labels:
                if correctStart and not isGold:
                    par = jobj['paragraph'] if 'paragraph' in jobj else jobj['text']
                    span = label['text'] if 'text' in label else label['span']

                    start = label['start']
                    end = label['end']

                    try:
                        # get the first matching span
                        for match in re.finditer(span, par):
                            start = match.start()
                            end = match.end()
                            break
                    except:
                        continue
                else:
                    start = label['start']
                    end = label['end']

                    if start == end == 0: continue
                per_par_labels.append((label['technique'], [start, end]))

            per_par_labels = sort_spans(per_par_labels)

            labels_per_par[par_id] = per_par_labels

    return labels_per_par


def compute_technique_frequency(annotations, technique_name):
    all_annotations = []
    for example_id, annot in annotations.items():
        for x in annot:
            all_annotations.append(x[0])

    techn_freq = sum([1 for a in all_annotations if a == technique_name])

    return techn_freq


def compute_span_score(gold_annots, pred_annots):
    prec_denominator = sum([len(pred_annots[x]) for x in pred_annots])
    rec_denominator = sum([len(gold_annots[x]) for x in gold_annots])

    technique_Spr_prec = {propaganda_technique: 0 for propaganda_technique in propaganda_techniques}
    technique_Spr_rec = {propaganda_technique: 0 for propaganda_technique in propaganda_techniques}
    cumulative_Spr_prec, cumulative_Spr_rec = (0, 0)
    f1_articles = []

    for example_id, pred_annot_obj in pred_annots.items():
        gold_annot_obj = gold_annots[example_id]

        document_cumulative_Spr_prec, document_cumulative_Spr_rec = (0, 0)
        for j, pred_ann in enumerate(pred_annot_obj):
            s = ""
            ann_length = pred_ann[1][1] - pred_ann[1][0]

            for i, gold_ann in enumerate(gold_annot_obj):
                if pred_ann[0] == gold_ann[0]:
                    intersection = span_intersection(gold_ann[1], pred_ann[1])
                    s_ann_length = gold_ann[1][1] - gold_ann[1][0]
                    Spr_prec = intersection / ann_length
                    document_cumulative_Spr_prec += Spr_prec
                    cumulative_Spr_prec += Spr_prec
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|p| = %d/%d = %f (cumulative S(p,r)=%f)\n" \
                         % (pred_ann[0], pred_ann[1][0], pred_ann[1][1], gold_ann[0],
                            gold_ann[1][0], gold_ann[1][1], intersection, ann_length, Spr_prec,
                            cumulative_Spr_prec)
                    technique_Spr_prec[gold_ann[0]] += Spr_prec

                    Spr_rec = intersection / s_ann_length
                    document_cumulative_Spr_rec += Spr_rec
                    cumulative_Spr_rec += Spr_rec
                    s += "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|r| = %d/%d = %f (cumulative S(p,r)=%f)\n" \
                         % (pred_ann[0], pred_ann[1][0], pred_ann[1][1], gold_ann[0],
                            gold_ann[1][0], gold_ann[1][1], intersection, s_ann_length, Spr_rec,
                            cumulative_Spr_rec)
                    technique_Spr_rec[gold_ann[0]] += Spr_rec

        p_article, r_article, f1_article = compute_prec_rec_f1(document_cumulative_Spr_prec,
                                                               len(pred_annot_obj),
                                                               document_cumulative_Spr_rec,
                                                               len(gold_annot_obj))
        f1_articles.append(f1_article)

    p, r, f1 = compute_prec_rec_f1(cumulative_Spr_prec, prec_denominator, cumulative_Spr_rec, rec_denominator)

    f1_per_technique = []

    for technique_name in technique_Spr_prec.keys():
        prec_tech, rec_tech, f1_tech = compute_prec_rec_f1(technique_Spr_prec[technique_name],
                                                           compute_technique_frequency(pred_annots,
                                                                                       technique_name),
                                                           technique_Spr_prec[technique_name],
                                                           compute_technique_frequency(gold_annots,
                                                                                       technique_name))
        f1_per_technique.append(f1_tech)

    return p, r, f1, f1_per_technique


def FLC_score_to_string(gold_annotations, user_annotations, per_label):
    precision, recall, f1, f1_per_class = compute_span_score(gold_annotations, user_annotations)

    if per_label:
        res_for_screen = f"\nF1=%f\nPrecision=%f\nRecall=%f\n%s\n" % (f1, precision, recall, "\n".join(
            ["F1_" + pr + "=" + str(f) for pr, f in
             zip(propaganda_techniques, f1_per_class)]))
    else:
        average = sum(f1_per_class) / len(f1_per_class)
        res_for_screen = f"Micro-F1\tMacro-F1\tPrecision\tRecall\n%f\t%f\t%f\t%f" % (f1, average, precision, recall)

    res_for_script = "%f\t%f\t%f\t" % (f1, precision, recall)
    res_for_script += "\t".join([str(x) for x in f1_per_class])

    return res_for_screen


def sort_spans(spans):
    """
    sort the list of annotations with respect to the starting offset
    """
    spans = sorted(spans, key=lambda span: span[1][0])

    return spans


def compute_prec_rec_f1(prec_numerator, prec_denominator, rec_numerator, rec_denominator):
    p, r, f1 = (0, 0, 0)
    if prec_denominator > 0:
        p = prec_numerator / prec_denominator
    if rec_denominator > 0:
        r = rec_numerator / rec_denominator
    if prec_denominator == 0 and rec_denominator == 0:
        f1 = 1.0
    if p > 0 and r > 0:
        f1 = 2 * (p * r / (p + r))

    return p, r, f1


def span_intersection(gold_span, pred_span):
    x = range(gold_span[0], gold_span[1])
    y = range(pred_span[0], pred_span[1])
    inter = set(x).intersection(y)
    return len(inter)


def eval_prop_predictions(gold_file, pred_file, correctStart, return_per_label_score):
    isGold = False
    user_annotations = load_json_as_list(pred_file, correctStart, isGold)

    isGold = True
    gold_annotations = load_json_as_list(gold_file, correctStart, isGold)

    res_for_output = FLC_score_to_string(gold_annotations, user_annotations, return_per_label_score)

    return res_for_output



if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-i', '--gold_file', action="store", dest="gold_file", default=None, type="string", help='gold annotations file.')
    parser.add_option('-o', '--pred_file', action="store", dest="pred_file", default=None, type='string', help="model annotations file.")

    options, args = parser.parse_args()

    correctStart = True
    return_per_label_score = False

    eval_prop_predictions(options.gold_file, options.pred_file, correctStart, return_per_label_score)