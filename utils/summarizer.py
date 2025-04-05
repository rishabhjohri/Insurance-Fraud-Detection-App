from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def make_prompt(row):
    return (
        f"{row['Claim_Description']}. "
        f"The customer is a {row['Customer_Age']}-year-old {row['Gender'].lower()} "
        f"{row['Marital_Status'].lower()} working as a {row['Occupation'].lower()}. "
        f"They have a policy with coverage of {row['Coverage_Amount']} and a premium of {row['Premium_Amount']}. "
        f"The claim history shows {int(row['Claim_History'])} previous claims."
    )

def generate_summary(text, max_input_length=512, max_output_length=100):
    input_text = "summarize: " + text.strip().replace("\\n", " ")
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    summary_ids = t5_model.generate(inputs, max_length=max_output_length, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
