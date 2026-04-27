package com.kotlintestapp.models
// @ast node: Class "Animal"
// @ast edge: ParentOf -> Class "Dog" "Person.kt"
// @ast node: Class "Dog"
// @ast node: Class "Person"
// @ast node: DataModel "Animal"
// @ast node: DataModel "Dog"
// @ast node: DataModel "Person"
// @ast node: Function "speak"
// @ast node: Function "speak"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappmodelspersonkt-0"

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
