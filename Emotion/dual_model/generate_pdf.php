<?php
require("C:/xampp/htdocs/dual_model/tcpdf_min/TCPDF-main/tcpdf.php"); // Include TCPDF library
$overallData = json_decode($_POST['overallData'], true);

$pdf = new TCPDF();
$pdf->AddPage();
$pdf->SetFont('helvetica', '', 12);
$pdf->Write(0, "Overall Emotion Report\n\n", '', 0, 'L', true, 0, false, false, 0);

if($overallData){
    $pdf->Write(0, "Overall Emotion: " . ($overallData['overall_emotion'] ?? '-') . "\n", '', 0, 'L', true, 0, false, false, 0);
    $pdf->Write(0, "Text Emotion: " . ($overallData['text_emotion'] ?? '-') . " (" . ($overallData['text_confidence'] ?? '-') . ")\n", '', 0, 'L', true, 0, false, false, 0);
    $pdf->Write(0, "Camera Emotion: " . ($overallData['camera_emotion'] ?? '-') . " (" . ($overallData['camera_confidence'] ?? '-') . ")\n", '', 0, 'L', true, 0, false, false, 0);
}

$pdf->Output('Emotion_Report.pdf', 'I');
?>
