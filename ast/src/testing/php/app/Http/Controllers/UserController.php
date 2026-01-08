<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Services\UserService;
use Illuminate\Http\Request;
use Illuminate\Http\RedirectResponse;
use Illuminate\View\View;

class UserController extends Controller
{
    public function __construct(
        protected UserService $userService
    ) {}

    /**
     * Display a listing of the resource.
     */
    public function index(): View
    {
        $users = User::paginate(10);
        return view('users.index', compact('users'));
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request): RedirectResponse
    {
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'email' => 'required|email|unique:users',
            'password' => 'required|min:8',
        ]);

        $this->userService->registerUser($validated);

        return redirect()->route('users.index')->with('success', 'User created.');
    }

    /**
     * Display the specified resource.
     */
    public function show(User $user): View
    {
        return view('users.show', compact('user'));
    }
}
