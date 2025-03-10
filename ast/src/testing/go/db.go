package main

import (
	"os"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

type database struct {
	db *gorm.DB
}

// DB is the object
var DB database

type Person struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Email string `json:"email"`
}

func (p *Person) TableName() string {
	return "people"
}

func (db database) CreatePerson(p Person) error {
	return db.db.Create(&p).Error
}

// check that update owner_pub_key does in fact throws an error
func (db database) CreateOrEditPerson(m Person) (Person, error) {
	if db.db.Model(&m).Where("id = ?", m.ID).Updates(&m).RowsAffected == 0 {
		db.db.Create(&m)
	}
	return m, nil
}

func (db database) UpdatePersonName(id uint, name string) {
	if id == 0 {
		return
	}
	db.db.Model(&Person{}).Where("id = ?", id).Updates(map[string]interface{}{
		"name": name,
	})
}

func (db database) GetPerson(id uint) (Person, error) {
	var p Person
	if err := db.db.First(&p, id).Error; err != nil {
		return Person{}, err
	}
	return p, nil
}

func InitDB() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		panic("DB env vars not found")
	}

	var err error

	db, err := gorm.Open(postgres.New(postgres.Config{
		DSN:                  dbURL,
		PreferSimpleProtocol: true,
	}), &gorm.Config{})

	if err != nil {
		panic(err)
	}

	DB.db = db

	db.AutoMigrate(&Person{})
}
