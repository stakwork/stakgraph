package com.kotlintestapp.models

data class Person(
    val id: Int,
    val owner_alias: String,
    val img: String,
    val owner_pubkey: String,
    val owner_route_hint: String
)

// For testing the Class - ParentOf -> Class edge
class Dog(name: String, val breed: String) : Animal(name) {
    override fun speak(): String = "Woof! I'm a $breed"
}
open class Animal(val name: String) {
    open fun speak(): String = "I am $name"
}
