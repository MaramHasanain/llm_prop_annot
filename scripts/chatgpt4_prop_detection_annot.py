import json
import os
import optparse
import openai
import ast
from dotenv import load_dotenv


from collections import defaultdict

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type



def input_chat(input_prompt, model):
    system_prompt = "You are an expert analyst of Arabic news articles.\n\n"

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{input_prompt}"},
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=messages,
        temperature=0,
        top_p=0.95,
        max_tokens=800,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response


def read_data(file_path, filtering_data):
    labels = defaultdict(list)
    pars = {}
    with open(file_path, mode='r', encoding="utf-8") as infile:
        for line in infile:
            jobj = json.loads(line)  # load 3 annotators
            for annot in jobj:
                annot = annot['info']
                parid = annot['parid']

                if parid not in filtering_data: continue

                # par = annot['paragraph']
                answers = json.loads(annot['answers'])
                pars[parid] = filtering_data[parid]
                for ans in answers:
                    labels[parid].append(ans)

    print("Number of examples: %d and those with labels: %d" % (len(pars), len(labels)))

    return pars, labels


def read_data_as_mlabel(file_path, filtering_data):
    labels = defaultdict(list)
    pars = {}
    with open(file_path, mode='r', encoding="utf-8") as infile:
        for line in infile:
            jobj = json.loads(line)  # load 3 annotators
            for annot in jobj:
                annot = annot['info']
                parid = annot['parid']

                if parid not in filtering_data: continue

                # par = annot['paragraph']
                answers = json.loads(annot['answers'])
                pars[parid] = filtering_data[parid]
                for ans in answers:
                    if 'technique' not in ans: continue
                    # print(ans)
                    labels[parid].append(ans['technique'])

    print("Number of examples: %d and those with labels: %d" % (len(pars), len(labels)))

    return pars, labels


def read_results_data(file_path):
    id_list = set()
    with open(file_path, 'r', encoding="utf-8") as json_file:
        for line in json_file:
            result = json.loads(line)
            id = str(result["paragraph_id"])
            id_list.add(id)
    print("Number of results item: {}".format(len(id_list)))
    return id_list


def filter_data(id_list, data):
    print("Number of items before: {}".format(len(data)))
    new_data = {}
    for data_id, row in data.items():
        if (data_id in id_list):
            continue
        else:
            new_data[data_id] = row

    print("Number of items after: {}".format(len(new_data)))

    return new_data


def fix_single_label(label):
    label = label.strip().lower()
    if "slogan" in label:
        label_fixed = "Slogans"
    if "loaded" in label:
        label_fixed = "Loaded_Language"
    if "prejudice" in label or "fear" in label or "mongering" in label:
        label_fixed = "Appeal_to_Fear-Prejudice"
    if "terminat" in label or "thought" in label or "conversation" in label or "killer" in label or "cliché" in label:
        label_fixed = "Conversation_Killer"
    if "calling" in label or label == "name c" or "labeling" in label:
        label_fixed = "Name_Calling-Labeling"
    if "minimisation" in label or label == "exaggeration minim" or "exaggeration" in label:
        label_fixed = "Exaggeration-Minimisation"
    if "values" in label or "virtue" in label or "generalities" in label or "glittering" in label:
        label_fixed = "Appeal_to_Values"
    if "flag" in label or "wav" in label:
        label_fixed = "Flag_Waving"
    if "obfusc" in label or "vague" in label or "confusion" in label:
        label_fixed = "Obfuscation-Vagueness-Confusion"
    if "causal" in label:
        label_fixed = "Causal_Oversimplification"
    if "conseq" in label:
        label_fixed = "Consequential_Oversimplification"
    if "authority" in label:
        label_fixed = "Appeal_to_Authority"
    if "choice" in label or "dilemma" in label or "false" in label or "dictatorship" in label or "black-and-white" in label or "black" in label:
        label_fixed = "False_Dilemma-No_Choice"
    if "herring" in label or "irrelevant" in label or "presenting" in label:
        label_fixed = "Red_Herring"
    if "straw" in label or "misrepresentation" in label or "someone's" in label:
        label_fixed = "Straw_Man"
    if "guilt" in label or "association" in label or "hitlerum" in label or "reductio" in label:
        label_fixed = "Guilt_by_Association"
    if "questioning" in label or "reputation" in label or "smear" in label:
        label_fixed = "Questioning_the_Reputation"
    if "whataboutism" in label:
        label_fixed = "Whataboutism"
    if "doubt" in label or "casting" in label:
        label_fixed = "Doubt"
    if "time" in label:
        label_fixed = "Appeal_to_Time"
    if "popularity" in label or "bandwagon" in label:
        label_fixed = "Appeal_to_Popularity"
    if "repetition" in label:
        label_fixed = "Repetition"
    if "hypocrisy" in label:
        label_fixed = "Appeal_to_Hypocrisy"

    if (
            "no propaganda" in label
            or "no technique" in label
            or label == ""
            or label == "no"
            or label == "none"
            or label == "appeal to history"
            or label == "appeal to emotion"
            or label == "appeal to"
            or label == "appeal"
            or label == "appeal to author"
            or label == "emotional appeal"
            or "no techn" in label
            or "hashtag" in label
            or "theory" in label
            or "specific mention" in label
            or "religious" in label
            or "gratitude" in label
    ):
        label_fixed = "no_technique"

    return label_fixed


def fix_span(prediction):
    prediction = prediction.replace("[\\n  ","[").replace("\\\n", "").replace("\'", "\"").strip()

    pred_labels = ast.literal_eval(prediction)

    for i, label in enumerate(pred_labels):
        if 'technique' not in label or 'start' not in label or 'end' not in label:
            print("Skipping %s with no technique" % label)
            print("len labels %d" % len(pred_labels))
            pred_labels.pop(i)
            print("len labels %d" % len(pred_labels))
            continue
        label['technique'] = label['technique'].strip().lower()
        label['technique'] = fix_single_label(label['technique'])

    final_labels = []
    for pred_label in pred_labels:
        if pred_label['technique'] != "no_technique":
            final_labels.append(pred_label)

    return final_labels


@retry(wait=wait_random_exponential(min=3, max=120), stop=stop_after_attempt(5000),
       retry=retry_if_not_exception_type((openai.InvalidRequestError, openai.error.Timeout)))
def api_call(pars, labels, output_file, err_output_file, model, base_prompt, wAnnots):
    k = 0
    for parid, row in pars.items():
        if k % 100 == 0 and k != 0: print("Done with %d rows so far..." % k)

        if parid not in labels:
            row['prediction'] = []
            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")

            continue

        sentence = row['paragraph']
        annots = labels[parid]

        if wAnnots:
            input_prompt = base_prompt + "Paragraph: " + sentence + "\n" + "Annotations: " + str(annots) + "\n\nResponse: "
        else:
            input_prompt = base_prompt + "Paragraph: " + sentence + "\n\n" + "Response: \n"

        try:
            response = input_chat(input_prompt, model)

            if response["choices"][0]["finish_reason"] == "content_filter":
                row['prediction'] = []
            else:
                response = response["choices"][0]["message"]["content"]
                row['prediction'] = fix_span(response)

            json_string = json.dumps(row, ensure_ascii=False)
            output_file.write(json_string + "\n")

        except Exception as e:
            row["error_msg"] = str(e)
            json_string = json.dumps(row, ensure_ascii=False)
            err_output_file.write(json_string + "\n")
            err_output_file.flush()

        k += 1


def continue_from_stopped(data, results_file_path):
    print("continuing from %s" % results_file_path)

    id_list = read_results_data(results_file_path)

    print("id_list: {}".format(len(id_list)))

    data_subset = filter_data(id_list, data)

    return data_subset


def safe_open(path, option):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)

    return open(path, option, encoding="utf-8")


def run_gpt_as_annotator(annots_fname, output_file, err_output_file, model):
    base_prompt = f'Label this \"Paragraph\" by the following propaganda techniques:\n\n' \
                  f"'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , 'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , 'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , 'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , 'Consequential Oversimplification' , 'False Dilemma/No Choice' , 'no technique'" \
                  f"\nAnswer exactly and only by returning a list of the matching labels from the aforementioned techniques and specify the start position and end position of the text span matching each technique. Use this template {{\"technique\": "" , \"text\": , \"start\": , \"end\": }}\n\n"


    filtering_data = continue_from_stopped(filtering_data, out_fname)
    pars, labels = read_data(annots_fname, filtering_data)

    api_call(pars, labels, output_file, err_output_file, model, base_prompt, False)

    output_file.close()
    err_output_file.close()



def run_gpt_as_selector(annots_fname, output_file, err_output_file, model):
    base_prompt = f'Given the following Paragraph and Annotations showing propaganda techniques potentially in it. Choose the techniques you are most confident appeared in Paragraph from all Annotations and return a Response.\n' \
                  f"Answer exactly and only by returning a list of the matching labels and specify the start position and end position of the text span matching each technique. Use this template {{\"technique\": "" , \"text\": , \"start\": , \"end\": }}"

    filtering_data = continue_from_stopped(filtering_data, out_fname)
    pars, labels = read_data_as_mlabel(annots_fname, filtering_data)

    api_call(pars, labels, output_file, err_output_file, model, base_prompt, True)

    output_file.close()
    err_output_file.close()


def run_gpt_as_consolidator(annots_fname, output_file, err_output_file, model):
    base_prompt = f'Given the following Paragraph and Annotations showing propaganda techniques potentially in it, and excerpt from the Paragraph where a technique is found. Choose the techniques you are most confident appeared in Paragraph from all Annotations and return a Response.\n' \
                  f"Answer exactly and only by returning a list of the matching annotations.\n\n"

    filtering_data = continue_from_stopped(filtering_data, out_fname)
    pars, labels = read_data(annots_fname, filtering_data)

    api_call(pars, labels, output_file, err_output_file, model, base_prompt, True)

    output_file.close()
    err_output_file.close()


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input_file', action="store", dest="input_file", default=None, type="string",
                      help='input human annotations file')
    parser.add_option('-o', '--output_file', action="store", dest="out_fname", default=None, type='string',
                      help="output file for model responses.")
    parser.add_option('-ef', '--err_output_file', action='store', dest='err_out_fname', default=None, type="string",
                      help='output file for failed model responses.')
    parser.add_option('-r', '--role', action='store', dest='role', default=None, type="string",
                      help='LLM annotation role: annot | select | cons')
    parser.add_option('-e', '--env', action='store', dest='env', default=None, type="string",
                      help='API key file')

    options, args = parser.parse_args()

    input_file = options.input_file
    output_file = safe_open(options.out_fname, 'a')
    err_output_file = safe_open(options.err_out_fname, 'a')

    # need to set openai api keys
    load_dotenv(options.env)
    openai.api_type = os.getenv('OPENAI_API_TYPE')
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_version = os.getenv('OPENAI_API_VERSION')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL')

    role = options.role
    if role == "annot":
        run_gpt_as_annotator(input_file, output_file, err_output_file, model)
    elif role == "select":
        run_gpt_as_selector(input_file, output_file, err_output_file, model)
    elif role == "cons":
        run_gpt_as_consolidator(input_file, output_file, err_output_file, model)
