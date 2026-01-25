
from playwright.sync_api import Page, expect

def test_homepage_has_title(page: Page):
    page.goto("https://my-python-app.com/")
    expect(page).to_have_title("Python App")

def test_create_user_flow(page: Page):
    page.goto("https://my-python-app.com/signup")
    page.get_by_label("Name").fill("David")
    page.get_by_label("Age").fill("28")
    page.get_by_role("button", name="Sign Up").click()
    
    expect(page.get_by_text("Welcome, David")).to_be_visible()

class TestUserFlows:
    def test_login_and_dashboard(self, page: Page):
        page.goto("https://my-python-app.com/login")
        page.fill("input[name='username']", "David")
        page.fill("input[name='password']", "securepassword")
        page.click("button[type='submit']")
        
        expect(page).to_have_url("https://my-python-app.com/dashboard")
