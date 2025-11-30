document.addEventListener("DOMContentLoaded", () => {
  const askBtn = document.getElementById("askBtn");
  const feedbackButtons = document.querySelectorAll(".feedback-btn");

  askBtn.addEventListener("click", async () => {
    const question = document.getElementById("question").value;
    const firstGuess = document.getElementById("first_guess").value;
    const onboardingPref = document.getElementById("onboarding_pref").value;

    const formData = new FormData();
    formData.append("question", question);
    formData.append("first_guess", firstGuess);
    formData.append("onboarding_pref", onboardingPref);

    const res = await fetch("/chat", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    const answerDiv = document.getElementById("answer");
    answerDiv.innerHTML = `
      <p>${data.answer || "(no answer)"}</p>
      <hr>
      <strong>Sources:</strong>
      <ul>
        ${(data.sources || []).map(s => `<li>${s}</li>`).join("")}
      </ul>
    `;
  });

  feedbackButtons.forEach(btn => {
    btn.addEventListener("click", async () => {
      const rating = btn.dataset.rating;
      const comment = document.getElementById("feedback_comment").value;
      const question = document.getElementById("question").value;
      const answer = document.getElementById("answer").innerText;

      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating,
          comment,
          question,
          answer
        })
      });

      alert("Thanks for your feedback!");
    });
  });
});
