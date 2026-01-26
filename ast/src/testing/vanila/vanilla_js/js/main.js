import { Navbar } from "./components/navbar.js";
import { Footer } from "./components/footer.js";
import { ApiClient } from "./api.js";
import { formatDate } from "./utils.js";

document.addEventListener("DOMContentLoaded", async () => {
  // Initialize common components
  const navbar = new Navbar();
  navbar.render();

  const footer = new Footer();
  footer.render();

  // Page specific logic
  const path = window.location.pathname;
  const api = new ApiClient();

  if (path.includes("index.html") || path === "/") {
    const updatesContainer = document.getElementById("updates-container");
    if (updatesContainer) {
      try {
        const updates = await api.get("/updates");
        updatesContainer.classList.remove("loading");
        updatesContainer.innerHTML = updates
          .map(
            (update) => `
                    <div class="update-item">
                        <h4>${update.title}</h4>
                        <small>${formatDate(update.date)}</small>
                    </div>
                `,
          )
          .join("");
      } catch (err) {
        updatesContainer.textContent = "Failed to load updates.";
      }
    }
  }

  if (path.includes("about.html")) {
    const teamContainer = document.getElementById("team-container");
    if (teamContainer) {
      try {
        const team = await api.get("/team");
        teamContainer.innerHTML = team
          .map(
            (member) => `
                    <div class="team-member card">
                        <h3>${member.name}</h3>
                        <p>${member.role}</p>
                    </div>
                `,
          )
          .join("");
      } catch (err) {
        teamContainer.textContent = "Failed to load team.";
      }
    }
  }

  if (path.includes("contact.html")) {
    const form = document.getElementById("contact-form");
    const feedback = document.getElementById("form-feedback");

    if (form) {
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
          const result = await api.post("/contact", data);
          feedback.textContent = result.message;
          feedback.classList.remove("hidden", "error");
          feedback.classList.add("success");
          form.reset();
        } catch (err) {
          feedback.textContent = "Something went wrong. Please try again.";
          feedback.classList.remove("hidden", "success");
          feedback.classList.add("error");
        }
      });
    }
  }
});
