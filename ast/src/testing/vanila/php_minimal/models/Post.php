<?php

require_once __DIR__ . '/../database.php';

class Post {
    public $id;
    public $title;
    public $content;
    public $user_id;

    public function __construct($title, $content, $user_id) {
        $this->title = $title;
        $this->content = $content;
        $this->user_id = $user_id;
    }

    public function save() {
        $db = Database::getInstance();
        $this->id = $db->insert('posts', [
            'title' => $this->title,
            'content' => $this->content,
            'user_id' => $this->user_id
        ]);
        return $this->id;
    }

    public static function all() {
        $db = Database::getInstance();
        return $db->findAll('posts');
    }
}
