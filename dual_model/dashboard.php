<?php
session_start();
error_reporting(0);
include('dbconnection.php'); 

 $login_id = $_SESSION['login_id'];

 $username="";
  $query1 = mysqli_query($con,"select * from tbl_user where id='$login_id'");
	while ($row1=mysqli_fetch_array($query1)) {
    $username = $row1['uname'];
  }
?>
  
  <!DOCTYPE html>
  <html lang="en">

  <head>
    <meta charset="utf-8">
    <meta content="width=device-width, initial-scale=1.0" name="viewport">

    <title>Dual-model</title>

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

    <style>
      .sidebar {
          width: 0px;
          left: -300px;
      }

      #main, #footer {
          margin-left: 0px;
      }

      #canvas {
      background-color: black;
      display: block;
      width: 30%;
      height: auto; /* Let JS handle actual height */
      }

      @media (max-width: 899px) {
        #canvas {
         width: 50%;
        }
      }

      @media (max-width: 599px) {
        #canvas {
         width: 70%;
        }
    }
    </style>

  </head>

  <body>

    <!-- ======= Header ======= -->
    <header id="header" class="header fixed-top d-flex align-items-center">

      <div class="d-flex align-items-center justify-content-between">
        <a href="index.html" class="logo d-flex align-items-center">
          <!-- <img src="assets/img/logo.png" alt=""> -->
          <span class="d-none d-lg-block">Dual-model</span>
        </a>
      </div><!-- End Logo -->

      <nav class="header-nav ms-auto">
      <ul class="d-flex align-items-center">

        <li class="nav-item dropdown pe-3"> <a class="p-0" href="dashboard.html"><i class="bi bi-house-fill fs-3"></i></a></li>
        <li class="nav-item dropdown pe-3">

          <a class="nav-link nav-profile d-flex align-items-center pe-0" href="#" data-bs-toggle="dropdown">
            <!-- <img src="assets/img/profile-img.jpg" alt="Profile" class="rounded-circle"> -->
            <span class="dropdown-toggle ps-2"><?php echo $username; ?></span>
          </a>

          <ul class="dropdown-menu dropdown-menu-end dropdown-menu-arrow profile">
            <li class="dropdown-header">
              <h6><?php echo $username; ?></h6>
            </li>
            <li>
              <hr class="dropdown-divider">
            </li>

            <li>
              <a class="dropdown-item d-flex align-items-center" href="logout.php">
                <i class="bi bi-box-arrow-right"></i>
                <span>Sign Out</span>
              </a>
            </li>

          </ul>
        </li><!-- End Profile Nav -->

      </ul>
    </nav><!-- End Icons Navigation -->

    </header><!-- End Header -->

    <main id="main" class="main">

      <div class="pagetitle">
        <h1 class="fs-2">Dashboard</h1>
      </div><!-- End Page Title -->

      <section class="section">
        <div class="row mt-4">
    
          <!-- Device Registration -->
          <div class="col-sm-12 col-lg-4">
            <a href="chatbot.php">
              <div class="card bg-primary" style="height: 200px;">
                <div class="card-body d-flex justify-content-center align-items-center mt-4">
                  <h3 class="text-white">Chatbot</h3>
                </div>
              </div>
            </a>
          </div><!-- End Device Registration   -->

          <!-- Device Configuration -->
          <div class="col-sm-12 col-lg-4">
            <a href="emotion_detection.php">
              <div class="card bg-primary" style="height: 100px;">
                <div class="card-body d-flex justify-content-center align-items-center mt-4">
                  <h3 class="text-white">Emotion Detection</h3>
                </div>
              </div>
            </a>
          </div><!-- End Device Configuration   -->

          <!-- Live Monitor -->
          <div class="col-sm-12 col-lg-4">
            <a href="quick_mood.php">
              <div class="card bg-primary" style="height: 100px;">
                <div class="card-body d-flex justify-content-center align-items-center mt-4">
                  <h3 class="text-white">Quick Mood Check</h3>
                </div>
              </div>
            </a>
          </div><!-- End Live Monitor -->

         
          <!-- <div class="col-sm-12 col-lg-3">
            <a href="report.php">
              <div class="card bg-primary" style="height: 100px;">
                <div class="card-body d-flex justify-content-center align-items-center mt-4">
                  <h3 class="text-white">Report</h3>
                </div>
              </div>
            </a>
          </div> -->


        </div>
      </section>

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
