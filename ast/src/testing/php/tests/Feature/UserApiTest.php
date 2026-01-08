<?php

use App\Models\User;

test('example', function () {
    expect(true)->toBeTrue();
});

test('user can be created', function () {
    $user = User::factory()->create();
    
    expect($user)->toBeInstanceOf(User::class);
});

it('has a welcome page', function () {
    $response = $this->get('/');
    
    $response->assertStatus(200);
});
