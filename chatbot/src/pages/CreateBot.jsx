import axios from "../api/axios";

export default function CreateBot() {
  const submit = async e => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));
    await axios.post("/bots", data);
    window.location.href = "/";
  };

  return (
    <div className="container create-page">
      <div className="page-header create-header">
        <h2 className="page-title">Create an AI Assistant</h2>
        <p className="page-subtitle">
          Set up a dedicated AI assistant for your organization.
          Each assistant has its own knowledge base and operates independently.
        </p>
      </div>

      <div className="card create-card">
        <form onSubmit={submit}>
          <div className="form-group">
            <label className="form-label">Assistant name</label>
            <input
              name="botName"
              placeholder="e.g. Customer Support Assistant"
              required
            />
            <p className="form-hint">
              This name is visible internally and helps identify the assistant.
            </p>
          </div>

          <div className="form-group">
            <label className="form-label">Primary use case</label>
            <select name="botPurpose" required>
              <option value="">Select a use case</option>
              <option>Customer Support</option>
              <option>Internal Knowledge Base</option>
              <option>Sales Assistant</option>
              <option>HR / Employee Onboarding</option>
              <option>IT Helpdesk</option>
              <option>Training & Education</option>
              <option>Healthcare Assistant</option>
              <option>Finance / Banking Assistant</option>
              <option>Other</option>
            </select>
            <p className="form-hint">
              This helps us tailor the assistant's behavior and defaults.
            </p>
          </div>

          <div className="form-group">
            <label className="form-label">Description (optional)</label>
            <input
              name="description"
              placeholder="Short description for internal reference"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Company website</label>
            <input
              name="websiteUrl"
              type="url"
              placeholder="https://www.example.com"
            />
            <p className="form-hint">
              Used to crawl your website for knowledge at ingestion time.
            </p>
          </div>

          <div className="form-group">
            <label className="form-label">Facebook page (optional)</label>
            <input
              name="facebookUrl"
              type="url"
              placeholder="https://www.facebook.com/yourpage"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Instagram profile (optional)</label>
            <input
              name="instagramUrl"
              type="url"
              placeholder="https://www.instagram.com/yourprofile"
            />
          </div>

          <div className="form-footer">
            <p className="form-hint">
              You can update settings and upload documents after creation.
              Assistants are private to your organization.
            </p>
            <button>Create assistant</button>
          </div>
        </form>
      </div>
    </div>
  );
}
