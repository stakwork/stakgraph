<?php
// @ast node: Class "BlogController"
// @ast node: Endpoint "/"
// @ast node: Endpoint "/posts/{id}"
// @ast node: Function "index"
// @ast node: Function "show"
// @ast node: Var "$message"

namespace SymfonyApp\Controller;

use Symfony.Bundle.FrameworkBundle.Controller.AbstractController;
use Symfony.Component.HttpFoundation.Response;
use Symfony.Component.Routing.Annotation.Route;
use SymfonyApp.Service.MessageGenerator;

#[Route('/blog', name: 'blog_')]
class BlogController extends AbstractController
{
    #[Route('/', name: 'index', methods: ['GET'])]
    public function index(MessageGenerator $messageGenerator): Response
    {
        $message = $messageGenerator->getHappyMessage();
        
        return $this->render('blog/index.html.twig', [
            'message' => $message,
        ]);
    }

    #[Route('/posts/{id}', name: 'show', requirements: ['id' => '\d+'])]
    public function show(int $id): Response
    {
        // ... query for post by $id
        
        return $this->json(['id' => $id, 'title' => 'Example Post']);
    }
}
