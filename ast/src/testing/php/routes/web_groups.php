<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\UserController;

Route::controller(UserController::class)->group(function () {
    Route::get('/users/group_index', 'index');
    Route::post('/users/group_store', 'store');
});
