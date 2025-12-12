require 'rails_helper'

feature "User Authentication" do
  scenario "user logs in successfully" do
    user = create(:person, email: "user@example.com", password: "password123")
    
    visit "/login"
    fill_in "Email", with: "user@example.com"
    fill_in "Password", with: "password123"
    click_button "Log In"
    
    expect(page).to have_content("Welcome")
  end

  scenario "user fails to log in with wrong password" do
    user = create(:person, email: "user@example.com", password: "password123")
    
    visit "/login"
    fill_in "Email", with: "user@example.com"
    fill_in "Password", with: "wrongpassword"
    click_button "Log In"
    
    expect(page).to have_content("Invalid credentials")
  end

  scenario "user logs out" do
    user = create(:person, email: "user@example.com", password: "password123")
    
    visit "/login"
    fill_in "Email", with: "user@example.com"
    fill_in "Password", with: "password123"
    click_button "Log In"
    click_button "Log Out"
    
    expect(page).to have_content("Logged out successfully")
  end
end
