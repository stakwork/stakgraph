<?php

namespace App\Services;

use App\Models\User;
use Illuminate\Support\Facades\Mail;
use App\Mail\WelcomeEmail;

class UserService
{
    public function __construct(
        protected UserRepository $users
    ) {}

    public function registerUser(array $data): User
    {
        $user = $this->users->create([
            'name' => $data['name'],
            'email' => $data['email'],
            'password' => bcrypt($data['password']),
        ]);

        Mail::to($user)->send(new WelcomeEmail($user));

        return $user;
    }

    public function getUser(int $id): ?User
    {
        return $this->users->find($id);
    }
}
