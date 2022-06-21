import re
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow_text as text

def get_idx_to_label_lookup_table(file_path):
    with open(file_path) as f:
        ner_tags_data = json.load(f)

        idx_to_label = {}
        for elem in ner_tags_data:
            idx_to_label[elem["language"]] = {v: k for k, v in elem["ner_tags"].items()}
        
        return idx_to_label

ner_tags_file_path = "ner_tags.json"
LANGUAGE_EN = "en"
LANGUAGE_RO = "ro"
# for splitting sentences in tokens
pattern = r"[\w']+|[.,!?;/)(]"
# used for parsing a dialog file
START_PERSONA_TOKEN = "<-- START PERSONA -->"
END_PERSONA_TOKEN = "<-- END PERSONA -->"
DESCRIPTION_TOKEN = "D:"
LABEL_TOKEN = "L:"
QUESTION_TOKEN = "Q:"
ANSWEAR_TOKEN = "A:"
# used to convert a label (int) predicted by the model to a label (string)
idx_to_label = get_idx_to_label_lookup_table(ner_tags_file_path)

def add_description_to_persona(persona, line):
    line = line[len(DESCRIPTION_TOKEN):].lstrip()
    key, value = line.split("=")
    if value.isdigit():
        value = int(value)

    description = persona["description"]
    if key not in description:
        description[key] = value
    else:
        if isinstance(description[key], list):
            description[key].append(value)
        else:
            description[key] = [description[key], value]
    return persona

def add_label_to_dialog(dialog, line):
    line = line[len(LABEL_TOKEN):].lstrip()
    dialog["label"] = line
    return dialog

def _add_utterance_to_dialog(dialog, line, token_type, key, language, model):
    utterance = line[len(token_type):].lstrip()
    
    if key not in dialog:
        dialog[key] = []
    dialog[key].append(utterance)

    tokens_key = f"tokenized_{key}"
    if tokens_key not in dialog:
        dialog[tokens_key] = []
    tokenized_utterance = re.findall(pattern, utterance)
    dialog[tokens_key].append(tokenized_utterance)

    ner_key = f"ner_{key}"
    if ner_key not in dialog:
        dialog[ner_key] = []
    probability_predictions = model.predict(tokenized_utterance)
    label_predictions = np.argmax(probability_predictions, axis=2)
    # it is a matrix of dimension (1, MAX_SEQUENCE) (MAX_SEQUNCE is imposed by the model)
    label_predictions = label_predictions[0]
    label_predictions = label_predictions[1:len(tokenized_utterance)]
    lookup_table = idx_to_label[language]
    labels = [lookup_table[idx] for idx in label_predictions]
    dialog[ner_key].append(labels)

    return dialog

def add_question_to_dialog(dialog, line, language, model):
    return _add_utterance_to_dialog(dialog, line, QUESTION_TOKEN, "questions", language, model)

def add_answear_to_dialog(dialog, line, language, model):
    return _add_utterance_to_dialog(dialog, line, ANSWEAR_TOKEN, "answears", language, model)

def preprocess_dialogs(file_path, model, language):
    personas = []
    dialog_entity = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if dialog_entity:
                    persona["dialog"].append(dialog_entity)
                    dialog_entity = {}
                continue
            elif line == START_PERSONA_TOKEN:
                persona = {
                    "description": {},
                    "dialog": []
                }
            elif line.startswith(DESCRIPTION_TOKEN):
                add_description_to_persona(persona, line)
            elif line.startswith(LABEL_TOKEN):
                add_label_to_dialog(dialog_entity, line)
            elif line.startswith(QUESTION_TOKEN):
                add_question_to_dialog(dialog_entity, line, language, model)
            elif line.startswith(ANSWEAR_TOKEN):
                add_answear_to_dialog(dialog_entity, line, language, model)
            elif line == END_PERSONA_TOKEN:
                personas.append(persona)
    return personas


if __name__ == "__main__":

    model = tf.keras.models.load_model("../models/ner_bert_en_uncased_L-4_H-256_A-4/model")
    input_dir_path = "dialogs"
    output_dir_path = "json"
    input_file_path = os.path.join(input_dir_path, "en_personas.txt")
    output_file_path = os.path.join(output_dir_path, "en_personas.json")
    personas = preprocess_dialogs(input_file_path, model, LANGUAGE_EN)
    with open(output_file_path, "w") as f:
        json.dump(personas, f)
    