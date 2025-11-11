<?php
session_start();
error_reporting(0);
$responses = $_POST['responses'] ?? '';
$overallData = null;

if ($responses) {
    $data = json_encode(['text' => $responses]);
    $ch = curl_init("http://127.0.0.1:5000/overall_emotion");
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    $result = curl_exec($ch);
    curl_close($ch);
    $overallData = json_decode($result, true);
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Overall Emotion Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
<h3>Overall Emotion Report</h3>

<?php if($overallData): ?>
    <ul class="list-group my-3">
        <li class="list-group-item"><b>Overall Emotion:</b> <?= $overallData['overall_emotion'] ?? '-' ?></li>
        <li class="list-group-item"><b>Text Emotion:</b> <?= $overallData['text_emotion'] ?? '-' ?> (<?= $overallData['text_confidence'] ?? '-' ?>)</li>
        <li class="list-group-item"><b>Camera Emotion:</b> <?= $overallData['camera_emotion'] ?? '-' ?> (<?= $overallData['camera_confidence'] ?? '-' ?>)</li>
    </ul>
    <form method="POST" action="generate_pdf.php">
        <input type="hidden" name="overallData" value='<?= json_encode($overallData) ?>'>
        <button class="btn btn-primary">Generate PDF</button>
    </form>
<?php else: ?>
    <p>No data available.</p>
<?php endif; ?>
</body>
</html>
