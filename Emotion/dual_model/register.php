<?php
session_start();
error_reporting(0);
include('dbconnection.php'); 

if (isset($_POST['register'])) {
    $uname = $_POST['uname'];
    $mail = $_POST['mail'];
    $pass = $_POST['pass'];
    $cpass = $_POST['cpass'];
    $hashedPass = password_hash($pass, PASSWORD_DEFAULT);
    // Fetch hashed password from DB
    if($pass==$cpass){
      $query = mysqli_query($con, "Insert into tbl_user (uname, mail, pass) values ('$uname', '$mail', '$hashedPass')");
      header('location:login.php');
    } else {
        $msg = "Invalid Details.";
    }
}
?>
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta content="width=device-width, initial-scale=1.0" name="viewport">

  <title>Dual-model Emotion Detection</title>
  <meta content="" name="description">
  <meta content="" name="keywords">

  <!-- Google Fonts -->
  <link href="https://fonts.gstatic.com" rel="preconnect">
  <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,300i,400,400i,600,600i,700,700i|Nunito:300,300i,400,400i,600,600i,700,700i|Poppins:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">

  <!-- Vendor CSS Files -->
  <link href="assets/vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">
  <link href="assets/vendor/bootstrap-icons/bootstrap-icons.css" rel="stylesheet">
  <link href="assets/vendor/boxicons/css/boxicons.min.css" rel="stylesheet">
  <link href="assets/vendor/quill/quill.snow.css" rel="stylesheet">
  <link href="assets/vendor/quill/quill.bubble.css" rel="stylesheet">
  <link href="assets/vendor/remixicon/remixicon.css" rel="stylesheet">
  <link href="assets/vendor/simple-datatables/style.css" rel="stylesheet">

  <!-- Template Main CSS File -->
  <link href="assets_1/css/style.css" rel="stylesheet">

</head>

<body>

  <main>
    <div class="container">

      <section class="section register min-vh-100 d-flex flex-column align-items-center justify-content-center py-4">
        <div class="container">
          <div class="row justify-content-center">
            <div class="col-lg-4 col-md-6 d-flex flex-column align-items-center justify-content-center">

              <div class="d-flex justify-content-center py-4">
                <a href="index.html" class="logo d-flex align-items-center w-auto">
                  <img src="assets/img/logo.png" alt="">
                  <span class="d-none d-lg-block">Dual-model Emotion Detection</span>
                </a>
              </div><!-- End Logo -->

              <div class="card mb-3">

                <div class="card-body">

                  <div class="pt-4 pb-2">
                    <h5 class="card-title text-center pb-0 fs-4">Create an Account</h5>
                    <h5 class="text-danger text-center"><?php echo $msg; ?></h5>
                  </div>

                  <form class="row g-3" method="POST">
                    <div class="col-12">
                      <label for="uname" class="form-label">Full Name</label>
                      <input type="text" name="uname" class="form-control" id="uname" onkeypress="return (event.charCode > 64 && event.charCode < 91) || (event.charCode > 96 && event.charCode < 123) || (event.charCode = 32);" required>
                    </div>

                    <div class="col-12">
                      <label for="mail" class="form-label">Email ID</label>
                      <input type="email" name="mail" class="form-control" id="mail" required>
                    </div>

                    <div class="col-12">
                      <label for="pass" class="form-label">Password</label>
                      <input type="password" name="pass" class="form-control" id="pass" required>
                    </div>

                    <div class="col-12">
                      <label for="cpass" class="form-label">Conform Password</label>
                      <input type="password" name="cpass" class="form-control" id="cpass" required>
                    </div>

                    <div class="col-12 mt-4">
                      <input type="submit" class="btn btn-primary w-100" name="register" value="Create Account">
                    </div>
                    <div class="col-12 text-center">
                      <p class="small mb-0">Already have an account? <a href="index.html">Log in</a></p>
                    </div>
                  </form>

                </div>
              </div>

            </div>
          </div>
        </div>

      </section>

    </div>
  </main><!-- End #main -->

  <a href="#" class="back-to-top d-flex align-items-center justify-content-center"><i class="bi bi-arrow-up-short"></i></a>

  <!-- Vendor JS Files -->
  <script src="assets/vendor/apexcharts/apexcharts.min.js"></script>
  <script src="assets/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
  <script src="assets/vendor/chart.js/chart.umd.js"></script>
  <script src="assets/vendor/echarts/echarts.min.js"></script>
  <script src="assets/vendor/quill/quill.js"></script>
  <script src="assets/vendor/simple-datatables/simple-datatables.js"></script>
  <script src="assets/vendor/tinymce/tinymce.min.js"></script>
  <script src="assets/vendor/php-email-form/validate.js"></script>

  <!-- Template Main JS File -->
  <script src="assets/js/main.js"></script>

</body>

</html>