<?php

function jsonResponse($data, $status = 200) {
    header('Content-Type: application/json');
    http_response_code($status);
    echo json_encode($data);
    exit;
}

function sanitize($input) {
    return htmlspecialchars(strip_tags(trim($input)));
}
