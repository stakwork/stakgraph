<?php
// @ast node: IntegrationTest ""
// @ast node: IntegrationTest ""
// @ast edge: Calls -> Function "create" "UserRepository.php"
// @ast node: IntegrationTest ""
// @ast node: IntegrationTest ""
// @ast node: IntegrationTest ""
// @ast node: Var "$response"
// @ast node: Var "$user"
// @ast node: Var "$response"
// @ast node: Var "$response"
// @ast node: Import "import-imports-srctestingphptestsfeatureuserapitestphp-13"

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

it('can list users', function () {
    $response = $this->getJson('/api/users');

    $response->assertStatus(200);
});

it('can create user via api', function () {
    $response = $this->postJson('/api/users', [
        'email' => 'test@example.com',
        'password' => 'password',
    ]);

    $response->assertStatus(201);
});
