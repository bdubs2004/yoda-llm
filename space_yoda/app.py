# app.py
from flask import Flask, render_template_string, request, jsonify
import sys
import os
import re

# Add src folder to Python path
SRC_DIR = os.path.join(os.path.dirname(__file__), "../src")
sys.path.append(SRC_DIR)

# Import functions from your inference module
from infer import search_faiss, generate_response, tokenizer, device, model


app = Flask(__name__)

# --------------------
# Helper: Build Yoda-style response
# --------------------
def get_response(user_message: str, side: str) -> str:
    try:
        # Retrieve top 3 context chunks
        retrieved_chunks = search_faiss(user_message, k=3)

        # Filter out example lines
        filtered_chunks = []
        for chunk in retrieved_chunks:
            lines = [line for line in chunk.splitlines() if not line.strip().startswith("Exercise:")]
            filtered_chunks.append(" ".join(lines))
        context = " ".join(filtered_chunks)

        # Fallback context
        if not context.strip():
            context = "Relevant Star Wars knowledge about Jedi, Sith, lightsabers, and the Force."

        # Build prompt with instructions at the top
        prompt = f"""You are Yoda, a wise Jedi Master.
Answer in Yoda-style speech.
Keep it short, concise, and in-character.
Use ONLY relevant Star Wars knowledge from the context below.

Context:
{context}

Question: {user_message}
Answer (Yoda-style):"""

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract Yoda answer only
        if "Answer (Yoda-style):" in full_output:
            response = full_output.split("Answer (Yoda-style):", 1)[1].strip()
        else:
            response = full_output.replace(prompt, '').strip()

        # Keep only first line for concise answer
        response = response.split("\n")[0].strip()
        response = response.strip('"')

        # Apply Sith tone
        if side.lower() == "sith":
            response = response.replace("Yoda", "Sith Lord").replace("wise", "dark")

        # Safety fallback
        if not response:
            response = "Silent, I am. Try again, you must."

        return response

    except Exception as e:
        print(f"[ERROR] {e}")
        return "Confused, I am. Hmm."

# --------------------
# HTML Template
# --------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Yoda Chat</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
<style>
body {
    font-family: 'Orbitron', sans-serif;
    margin:0; padding:0;
    background: radial-gradient(circle at center, #0f2027, #203a43, #2c5364);
    color: #00ffcc;
    transition: all 0.5s ease;
    overflow-x: hidden;
    cursor: url('https://cur.cursors-4u.net/movie/mov-1/mov29.ani'),
           url('https://cur.cursors-4u.net/movie/mov-1/mov29.png'),
           auto !important;
}

body.sith {
    background: radial-gradient(circle at center, #1a0000, #4d0000, #0a0000);
    color: #ff0044;
}

header {
    text-align:center;
    padding:20px;
    font-size:2em;
    text-shadow:0 0 15px #00ffcc;
    transition: all 0.5s ease;
}

#container {
    max-width:800px;
    margin: 0 auto;
    padding: 20px;
    border-radius:20px;
    transition: all 0.5s ease;
    height: 80vh;
    display:flex;
    flex-direction: column;
    position: relative;
    z-index: 1;
}

#chatbox {
    flex:1;
    overflow-y:auto;
    padding:10px;
    border-radius:10px;
    background: rgba(0,0,0,0.85);
    margin-bottom:10px;
    display:flex;
    flex-direction: column;
    gap: 10px;
}

.message { padding:10px 15px; border-radius:12px; font-size:1.1em; line-height:1.4; max-width:70%; word-wrap: break-word; transition: all 0.3s ease;}
.message.user { align-self:flex-end; }
.message.bot { align-self:flex-start; }

.message.user.jedi, .message.bot.jedi { background: rgba(0,255,200,0.1); border:1px solid #00ffcc; box-shadow:0 0 10px rgba(0,255,200,0.4);}
.message.user.sith, .message.bot.sith { background: rgba(255,0,70,0.1); border:1px solid #ff0044; box-shadow:0 0 10px rgba(255,0,70,0.5); }

#controls { display:flex; gap:10px; align-items:center; }
#txt { flex:1; font-family:'Orbitron',sans-serif; padding:10px; font-size:1.1em; border-radius:8px; border:none; background: rgba(0,0,0,0.7); color:inherit;}
#send-btn { padding:10px 20px; font-size:1.1em; font-family:'Orbitron',sans-serif; border-radius:8px; border:none; cursor:pointer; transition: all 0.3s ease; }
body.jedi #send-btn { background:#00ffcc; color:#000; box-shadow:0 0 15px #00ffcc; }
body.sith #send-btn { background:#ff0044; color:#000; box-shadow:0 0 15px #ff0044; }
select { padding:10px; border-radius:8px; font-family:'Orbitron',sans-serif; font-size:1em; border:none; background: rgba(0,0,0,0.7); color: inherit; text-shadow:0 0 5px currentColor; }

#lightsaber-left, #lightsaber-right {
    position:absolute;
    top:50%;
    width:20px;
    height:300px;
    border-radius:10px;
    z-index: 0;
    opacity:0.9;
    transition: all 1s ease;
}

#lightsaber-left.jedi { background: linear-gradient(to top,#00ff00,#00ffcc); box-shadow:0 0 30px #00ffcc; left:-200px; transform: translateY(-50%);}
#lightsaber-right.jedi { background: linear-gradient(to top,#00ff00,#00ffcc); box-shadow:0 0 30px #00ffcc; right:-200px; transform: translateY(-50%);}
#lightsaber-left.sith { background: linear-gradient(to top,#ff0044,#ff3399); box-shadow:0 0 30px #ff0044; left:-200px; transform: translateY(-50%);}
#lightsaber-right.sith { background: linear-gradient(to top,#ff0044,#ff3399); box-shadow:0 0 30px #ff0044; right:-200px; transform: translateY(-50%);}

#lightsaber-left.animate { left:50%; transform: translate(-50%,-50%); }
#lightsaber-right.animate { right:50%; transform: translate(50%,-50%); }
</style>
</head>
<body class="jedi">
<header>Talk to Yoda</header>
<div id="container">
    <div id="lightsaber-left" class="jedi"></div>
    <div id="lightsaber-right" class="jedi"></div>
    <div id="chatbox"></div>
    <div id="controls">
        <input type="text" id="txt" placeholder="Ask Yoda anything..." />
        <button id="send-btn">Send</button>
        <select id="side">
            <option value="Jedi">Jedi</option>
            <option value="Sith">Sith</option>
        </select>
    </div>
</div>
<script>
const chatbox = document.getElementById('chatbox');
const txt = document.getElementById('txt');
const sendBtn = document.getElementById('send-btn');
const sideSelect = document.getElementById('side');
const lightsaberLeft = document.getElementById('lightsaber-left');
const lightsaberRight = document.getElementById('lightsaber-right');
const header = document.querySelector('header');

function addMessage(sender,text,side){
    const div = document.createElement('div');
    div.classList.add('message',sender,side==='Sith'?'sith':'jedi');
    div.textContent=text;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

async function sendMessage(){
    const message = txt.value.trim();
    if(!message) return;
    const side = sideSelect.value;

    // Add user message immediately
    addMessage('user', message, side);

    // Add thinking placeholder
    const thinkingDiv = document.createElement('div');
    thinkingDiv.classList.add('message','bot', side==='Sith'?'sith':'jedi');
    thinkingDiv.textContent = "Yoda is thinking...";
    chatbox.appendChild(thinkingDiv);
    chatbox.scrollTop = chatbox.scrollHeight;

    txt.value='';  // clear input box

    try {
        const res = await fetch('/ask',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({message,side})
        });
        const data = await res.json();

        // Replace placeholder with actual response
        thinkingDiv.textContent = data.reply;

    } catch(err) {
        thinkingDiv.textContent = "Confused, I am. Hmm.";
        console.error(err);
    }
}

function animateLightsabers(){
    lightsaberLeft.classList.add('animate');
    lightsaberRight.classList.add('animate');
    setTimeout(()=>{
        lightsaberLeft.classList.remove('animate');
        lightsaberRight.classList.remove('animate');
    },1000);
}

txt.addEventListener('keydown', e => { if(e.key==='Enter') sendMessage(); });
sendBtn.addEventListener('click', sendMessage);

sideSelect.addEventListener('change', function(){
    const side = sideSelect.value;
    document.body.className = side.toLowerCase();

    if(side==='Jedi'){
        sendBtn.style.background='#00ffcc';
        sendBtn.style.boxShadow='0 0 15px #00ffcc';
        header.style.textShadow='0 0 15px #00ffcc';
        lightsaberLeft.className='jedi';
        lightsaberRight.className='jedi';
    } else {
        sendBtn.style.background='#ff0044';
        sendBtn.style.boxShadow='0 0 15px #ff0044';
        header.style.textShadow='0 0 15px #ff0044';
        lightsaberLeft.className='sith';
        lightsaberRight.className='sith';
    }

    animateLightsabers();

    document.querySelectorAll('.message').forEach(m=>{
        m.classList.remove('jedi','sith');
        m.classList.add(side==='Sith'?'sith':'jedi');
    });
});

// Run lightsaber animation once on page load
window.addEventListener('DOMContentLoaded', () => {
    animateLightsabers();
});
</script>
</body>
</html>
"""

# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    message = data.get("message", "")
    side = data.get("side", "Jedi")
    reply = get_response(message, side)
    return jsonify({"reply": reply})

# --------------------
# Run server
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
