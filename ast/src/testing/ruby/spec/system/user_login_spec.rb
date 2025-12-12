# E2E/system (Capybara)
RSpec.describe "User login", type: :system, js: true do
  let(:user) { create(:person, email: "user@example.com", password: "password123") }
  
  before do
    driven_by(:selenium_chrome_headless)
  end
  
  describe "successful login" do
    it "allows user to log in with valid credentials" do
      visit "/login"
      
      within("#login-form") do
        fill_in "Email", with: user.email
        fill_in "Password", with: "password123"
        click_button "Login"
      end
      
      expect(page).to have_current_path("/dashboard")
      expect(page).to have_content("Welcome back")
      expect(page).to have_selector(".user-name", text: user.name)
    end
    
    it "displays user dashboard with personalized content" do
      person_with_articles = create(:person, :with_articles, email: "author@example.com", password: "pass123")
      
      visit "/login"
      
      within("#login-form") do
        fill_in "Email", with: "author@example.com"
        fill_in "Password", with: "pass123"
        click_button "Login"
      end
      
      expect(page).to have_content("Your Articles")
      expect(page).to have_selector(".article-item", count: 2)
    end
    
    it "remembers user session across page navigations" do
      visit "/login"
      
      within("#login-form") do
        fill_in "Email", with: user.email
        fill_in "Password", with: "password123"
        check "Remember me"
        click_button "Login"
      end
      
      visit "/profile"
      expect(page).to have_content(user.name)
      expect(page).not_to have_content("Please log in")
    end
  end
  
  describe "failed login attempts" do
    it "shows error message with invalid email" do
      visit "/login"
      
      within("#login-form") do
        fill_in "Email", with: "nonexistent@example.com"
        fill_in "Password", with: "wrongpassword"
        click_button "Login"
      end
      
      expect(page).to have_current_path("/login")
      expect(page).to have_selector(".alert-error", text: "Invalid email or password")
    end
    
    it "shows error message with incorrect password" do
      visit "/login"
      
      within("#login-form") do
        fill_in "Email", with: user.email
        fill_in "Password", with: "wrongpassword"
        click_button "Login"
      end
      
      expect(page).to have_selector(".alert-error", text: "Invalid email or password")
      expect(page).to have_field("Email", with: user.email)
    end
    
    it "shows validation errors for empty fields" do
      visit "/login"
      
      within("#login-form") do
        click_button "Login"
      end
      
      expect(page).to have_content("Email can't be blank")
      expect(page).to have_content("Password can't be blank")
    end
  end
  
  describe "logout functionality" do
    before do
      visit "/login"
      
      within("#login-form") do
        fill_in "Email", with: user.email
        fill_in "Password", with: "password123"
        click_button "Login"
      end
    end
    
    it "logs out user and redirects to login page" do
      within(".user-menu") do
        click_link "Logout"
      end
      
      expect(page).to have_current_path("/login")
      expect(page).to have_content("You have been logged out")
    end
    
    it "clears session after logout" do
      within(".user-menu") do
        click_link "Logout"
      end
      
      visit "/dashboard"
      expect(page).to have_current_path("/login")
      expect(page).to have_content("Please log in to continue")
    end
  end
  
  describe "password reset workflow" do
    it "allows user to request password reset" do
      visit "/login"
      click_link "Forgot password?"
      
      expect(page).to have_current_path("/password/reset")
      
      within("#reset-form") do
        fill_in "Email", with: user.email
        click_button "Send reset instructions"
      end
      
      expect(page).to have_content("Password reset instructions sent")
    end
  end
  
  describe "social login options" do
    it "displays social login buttons" do
      visit "/login"
      
      expect(page).to have_button("Login with Google")
      expect(page).to have_button("Login with GitHub")
    end
    
    it "shows login form as primary option" do
      visit "/login"
      
      within("#login-form") do
        expect(page).to have_field("Email")
        expect(page).to have_field("Password")
      end
    end
  end
end
