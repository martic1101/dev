<?php
    $flag = "ADLCTF(????????)"; // TODO
    $username = isset($_GET['username']) ? $_GET['username'] : '';
    $password = isset($_GET['password']) ? $_GET['password'] : '';
    
    if ($username === '' && $password === ''){
        header("Location: ./index.html" );
    }
    else {
        var_dump(md5($password));
        if(md5($password)=="0e481756596645574257920728035178" && !strcmp($username,$myflag)){
            include_once('flag.php');
            echo "your flag is " . $myflag;
        }
        else{
            $text = "Yee.";
            echo $text;
        }
    }
?>
