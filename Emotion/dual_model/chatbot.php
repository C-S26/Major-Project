<?php 
session_start();
error_reporting(0);
include('dbconnection.php'); 
$login_id = $_SESSION['login_id'];
$username = "";
$query1 = mysqli_query($con, "SELECT * FROM tbl_user WHERE id='$login_id'");
while ($row1 = mysqli_fetch_array($query1)) {
  $username = $row1['uname'];
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chatbot & Camera</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body { background-color: #f5f5f5; min-height: 100vh; display: flex; flex-direction: column; }
.chat-container { background-color: white; border-radius: 10px; height: 75vh; overflow-y: auto; padding: 15px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
.chat-message { margin-bottom: 12px; }
.user-msg { text-align: right; color: white; background-color: #0d6efd; display: inline-block; padding: 10px 15px; border-radius: 15px; max-width: 75%; word-wrap: break-word; }
.bot-msg { text-align: left; background-color: #e9ecef; display: inline-block; padding: 10px 15px; border-radius: 15px; max-width: 75%; word-wrap: break-word; }
.input-section { margin-top: 15px; }
.camera-container { position: relative; background-color: black; border-radius: 10px; height: 250px; display: flex; justify-content: center; align-items: center; overflow: hidden; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-top: 40px; }
#camera-canvas { width: 100%; height: 100%; border-radius: 10px; object-fit: cover; }
#overall-btn { margin-top: 10px; display: inline-block; }
</style>
</head>
<body>
<div class="container-fluid p-4">
  <div class="row">
    <!-- Chat Section -->
    <div class="col-md-8">
      <h4 class="mb-3">Chatbot</h4>
      <div class="chat-container" id="chat-box">
        <div class="chat-message"><span class="bot-msg">Hello <?php echo $username; ?>! Let's start with a few questions.</span></div>
      </div>
      <div class="input-section d-flex">
        <input type="text" id="user-input" class="form-control me-2" placeholder="Type your answer..." autocomplete="off">
        <button class="btn btn-primary" onclick="sendMessage()">Send</button>
      </div>
      <button class="btn btn-success" id="overall-btn" onclick="goToOverallPage()">Show Overall Emotion</button>
    </div>

    <!-- Camera Section -->
    <div class="col-md-4">
      <h4 class="mb-3">Camera Feed</h4>
      <div class="camera-container">
        <canvas id="camera-canvas"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
const questions = [
  "How are you feeling right now?",
  "Can you describe a recent moment that made you feel happy or content?",
  "Have you felt sad or down in the past 24 hours? If so, what happened?",
  "Did anything make you feel anxious or stressed recently?",
  "Can you recall a time today when you felt angry or frustrated?",
  "Was there a recent event that surprised you or caught you off guard?",
  "Have you experienced any fear or worry that lasted for a while?",
  "On a scale of 1 to 10, how relaxed or calm do you feel right now?",
  "Can you share a small win or positive moment from today?",
  "If you could change one thing about your day to feel better, what would it be?"
];

let currentQuestion = 0;
let userResponses = [];
const chatBox = document.getElementById('chat-box');

function addMessage(text, cls) {
  const msg = document.createElement('div');
  msg.classList.add('chat-message');
  msg.innerHTML = `<span class="${cls}">${text}</span>`;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function askQuestion() {
  if (currentQuestion < questions.length) {
    addMessage(questions[currentQuestion], 'bot-msg');
  }
}

askQuestion();

function sendMessage() {
  const input = document.getElementById('user-input');
  let message = input.value.trim();
  if (!message) return;

  addMessage(message, 'user-msg');
  userResponses.push(message);
  input.value = '';
  message = message.replace(/\s+/g, ' ');

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: message })
  })
  .then(res => res.json())
  .then(data => {
    if (data.emotion) {
      addMessage(`Detected emotion: <b>${data.emotion}</b> (Confidence: ${data.confidence})`, 'bot-msg');
    }

    currentQuestion++;
    if (currentQuestion < questions.length) {
      setTimeout(askQuestion, 800);
    } else {
      setTimeout(() => addMessage("Thank you for answering all questions!", 'bot-msg'), 800);
    }
  })
  .catch(err => {
    console.error("Error:", err);
    addMessage("⚠️ Unable to connect to Flask API.", 'bot-msg');
  });
}

function goToOverallPage() {
    const overallText = userResponses.join(" ");
    const form = document.createElement("form");
    form.method = "POST";
    form.action = "overall.php";
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "responses";
    input.value = overallText;
    form.appendChild(input);
    document.body.appendChild(form);
    form.submit();
}

// Camera canvas for MJPEG + overlay
function startCameraCanvas() {
  const canvas = document.getElementById("camera-canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();

  function drawFrame() {
    img.src = "http://127.0.0.1:5000/camera_feed?" + new Date().getTime();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      requestAnimationFrame(drawFrame);
    }
  }

  drawFrame();
}

startCameraCanvas();
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
