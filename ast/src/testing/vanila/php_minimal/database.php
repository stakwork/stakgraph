<?php

require_once 'config.php';

class Database {
    private static $instance = null;
    private $store = [];

    private function __construct() {
        // Simulating a connection
        $this->store['users'] = [];
        $this->store['posts'] = [];
    }

    public static function getInstance() {
        if (self::$instance == null) {
            self::$instance = new Database();
        }
        return self::$instance;
    }

    public function insert($table, $data) {
        $id = uniqid();
        $data['id'] = $id;
        $this->store[$table][$id] = $data;
        return $id;
    }

    public function find($table, $id) {
        return isset($this->store[$table][$id]) ? $this->store[$table][$id] : null;
    }

    public function findAll($table) {
        return array_values($this->store[$table]);
    }
}
