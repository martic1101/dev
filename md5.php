<?php
    $flag = "ADLCTF(????????)"; // TODO
    $username = isset($_GET['username']) ? $_GET['username'] : '';
    $password = isset($_GET['password']) ? $_GET['password'] : '';
    
    if ($username === '' && $password === ''){
        header("Location: ./index.html" );
    }
    else {
        var_dump(md5($password));
        if(password_hash($password)=="0e481756596645574257920728035178" && !strcmp($username,$myflag)){
            use flag.php;
            echo "your flag is " . $myflag;
        }
        else{
            $text = "Yee.";
            echo $text;
        }
    }
?>
