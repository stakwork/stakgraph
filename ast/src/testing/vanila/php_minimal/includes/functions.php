<?php

function generateSlug($string) {
    return strtolower(trim(preg_replace('/[^A-Za-z0-9-]+/', '-', $string)));
}

function encryptPassword($password) {
    return password_hash($password, PASSWORD_BCRYPT);
}
