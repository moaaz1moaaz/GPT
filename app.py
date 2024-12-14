from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# تحميل النموذج والمحسن (tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    
    # تحويل المدخلات إلى رموز يمكن للنموذج فهمها
    inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt')
    
    # توليد الرد باستخدام النموذج
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # تحويل الرموز الناتجة إلى نص قابل للقراءة
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # إرجاع الرد في شكل JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
