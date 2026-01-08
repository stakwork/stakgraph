<?php

namespace SymfonyApp\Entity;

use Doctrine.ORM.Mapping as ORM;
use SymfonyApp.Repository.ProductRepository;

#[ORM.Entity(repositoryClass: ProductRepository::class)]
#[ORM.Table(name: 'products')]
class Product
{
    #[ORM.Id]
    #[ORM.GeneratedValue]
    #[ORM.Column(type: 'integer')]
    private ?int $id = null;

    #[ORM.Column(type: 'string', length: 255)]
    private ?string $name = null;

    #[ORM.Column(type: 'integer')]
    private ?int $price = null;

    #[ORM.Column(type: 'text', nullable: true)]
    private ?string $description = null;

    public function getId(): ?int
    {
        return $this->id;
    }

    public function getName(): ?string
    {
        return $this->name;
    }

    public function setName(string $name): self
    {
        $this->name = $name;

        return $this;
    }
}
