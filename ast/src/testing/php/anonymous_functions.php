<?php

use Illuminate\Support\Facades\Route;
use App\Models\User;
use App\Models\Post;

// Laravel closure routes
Route::get('/users-closure', function() {
    return User::all();
});

Route::post('/posts-closure/{id}/like', function($id) {
    return Post::find($id)->like();
});

// Arrow function routes
Route::get('/status-arrow', fn() => ['status' => 'ok']);
