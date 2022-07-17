import json
import os

out_root_dir = "text_classification"
out_en_patient_path = os.path.join(out_root_dir, "en.json")
out_ro_patient_path = os.path.join(out_root_dir, "ro.json")

wit_ai_en_utterances = os.path.join("raw_wit_ai_data", "pav_en", "utterances", "utterances-1.json")
wit_ai_ro_utterances = os.path.join("raw_wit_ai_data", "pav", "utterances", "utterances-1.json")

input_files = [
    wit_ai_en_utterances,
    wit_ai_ro_utterances
]

output_files = [
    out_en_patient_path,
    out_ro_patient_path
]

lookup_table = {
    "chestpain": "chest_pain",
    "chronictreatment": "chronic_treatment",
    "coughsymp": "cough_symptoms",
    "diseasesparents": "diseases_parents",
    "diseasespersonal": "diseases_personal",
    "faintingsymp": "fainting_symptoms",
    "feversymp": "fever_symptoms",
    "greetings": "greetings",
    "palpitationssymp": "palpitations_symptoms",
    "surgeries": "surgeries",
    "symptomschanges": "symptoms_changes",
    "symptomscircumstances": "symptoms_circumstances",
    "symptomsstart": "symptoms_start",
    "visitreason": "visit_reason"
}

def preprocess(input_file, output_file):
    with open(input_file, encoding="utf-8") as infile:
        data = json.load(infile)
        
        utterances = []
        for utterance in data["utterances"]:
            if not "intent" in utterance or not "text" in utterance:
                continue
            utterances.append({
                "text": utterance["text"],
                "label": lookup_table[utterance["intent"]]
            })
        
        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(utterances, outfile)

if __name__ == "__main__":
    if not os.path.isdir(out_root_dir):
        os.mkdir(out_root_dir)
    
    for input_file, output_file in zip(input_files, output_files):
        preprocess(input_file, output_file)

