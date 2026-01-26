<?php

require_once 'config.php';
require_once 'controllers/UserController.php';
require_once 'controllers/PostController.php';

$uri = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
$method = $_SERVER['REQUEST_METHOD'];

// Simple routing
if ($uri === '/users' && $method === 'GET') {
    (new UserController())->index();
} elseif ($uri === '/users' && $method === 'POST') {
    (new UserController())->create();
} elseif (preg_match('/^\/users\/([a-z0-9]+)$/', $uri, $matches) && $method === 'GET') {
    (new UserController())->show($matches[1]);
} elseif ($uri === '/posts' && $method === 'GET') {
    (new PostController())->index();
} elseif ($uri === '/posts' && $method === 'POST') {
    (new PostController())->create();
} else {
    header("HTTP/1.0 404 Not Found");
    echo json_encode(['error' => 'Not Found']);
}
