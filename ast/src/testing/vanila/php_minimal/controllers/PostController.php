<?php

require_once __DIR__ . '/../models/Post.php';
require_once __DIR__ . '/../includes/helpers.php';

class PostController {
    public function index() {
        $posts = Post::all();
        jsonResponse($posts);
    }

    public function create() {
        $data = json_decode(file_get_contents('php://input'), true);
        
        $post = new Post(
            sanitize($data['title']), 
            sanitize($data['content']), 
            $data['user_id']
        );
        $post->save();
        
        jsonResponse(['id' => $post->id, 'message' => 'Post created'], 201);
    }
}
