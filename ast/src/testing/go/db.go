package main

// @ast node: Class "database"
// @ast edge: Operand -> Function "GetPersonById" "db.go"
// @ast edge: Operand -> Function "NewPerson" "db.go"
// @ast edge: Operand -> Function "GetPeopleLeaderboard" "db.go"
// @ast edge: Operand -> Function "CreateOrEditPerson" "db.go"
// @ast edge: Operand -> Function "UpdatePersonName" "db.go"
// @ast node: DataModel "database"
// @ast node: Class "Person"
// @ast edge: Operand -> Function "TableName" "db.go"
// @ast node: DataModel "Person"
// @ast node: DataModel "LeaderboardEntry"
// @ast node: Class "bountyHandler"
// @ast edge: Operand -> Function "GetBountiesLeaderboard" "db.go"
// @ast node: DataModel "bountyHandler"
// @ast node: Class "bountyDB"
// @ast edge: Operand -> Function "GetBountiesLeaderboard" "db.go"
// @ast node: DataModel "bountyDB"
// @ast node: Var "DB"
// @ast node: Instance "DB"
// @ast edge: Of -> Class "database" "db.go"
// @ast node: Function "GetBountiesLeaderboard"
// @ast node: Function "GetBountiesLeaderboard"
// @ast node: Function "GetPeopleLeaderboard"
// @ast node: Function "GetLeaderboard"
// @ast edge: Calls -> Function "GetPeopleLeaderboard" "db.go"
// @ast node: Function "TableName"
// @ast node: Function "NewPerson"
// @ast node: Function "CreateOrEditPerson"
// @ast node: Function "UpdatePersonName"
// @ast node: Function "GetPersonById"
// @ast node: Function "InitDB"

import (
	"encoding/json"
	"net/http"
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

type LeaderboardEntry struct {
    Name  string `json:"name"`
    Score int    `json:"score"`
}

type bountyHandler struct {
    db *bountyDB
}

type bountyDB struct{}

func (db *bountyDB) GetBountiesLeaderboard() []LeaderboardEntry {
    return []LeaderboardEntry{
        {Name: "Carol", Score: 200},
        {Name: "Dave", Score: 180},
    }
}

func (db database) GetPeopleLeaderboard() []LeaderboardEntry {
    return []LeaderboardEntry{
        {Name: "Alice", Score: 100},
        {Name: "Bob", Score: 90},
    }
}

func (h *bountyHandler) GetBountiesLeaderboard(w http.ResponseWriter, _ *http.Request) {
    leaderBoard := h.db.GetBountiesLeaderboard()
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(leaderBoard)
}

func GetLeaderboard(w http.ResponseWriter, r *http.Request) {
    leaderboard := DB.GetPeopleLeaderboard()
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(leaderboard)
}


func (p *Person) TableName() string {
	return "people"
}

func (db database) NewPerson(p Person) error {
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

func (db database) GetPersonById(id uint) (Person, error) {
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
