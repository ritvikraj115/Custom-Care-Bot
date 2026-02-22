import axios from "../api/axios";
import { toast } from "react-toastify";
import { getErrorMessage } from "../utils/getErrorMessage";

export default function Register() {
  const submit = async e => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));

    try {
      await axios.post("/auth/register", data);
      toast.success("Account created successfully. Please sign in.");
      window.location.href = "/login";
    } catch (error) {
      const status = error.response?.status;

      if (status === 400) {
        toast.error(getErrorMessage(error, "Invalid input or email already exists"));
      } else {
        toast.error(getErrorMessage(error, "Unable to create account"));
      }
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-layout">
        <div className="page-header">
          <h2 className="page-title">Create your organization</h2>
          <p className="page-subtitle">
            Set up a secure workspace for managing AI assistants.
          </p>
        </div>

        <div className="card auth-card">
          <form onSubmit={submit}>
            <label className="form-label">Organization name</label>
            <input
              name="companyName"
              placeholder="e.g. Acme Corporation"
              required
            />

            <label className="form-label">Work email</label>
            <input
              name="email"
              placeholder="you@company.com"
              required
            />

            <label className="form-label">Password</label>
            <input
              name="password"
              type="password"
              placeholder="Create a secure password"
              required
            />

            <label className="form-label">Industry (optional)</label>
            <input
              name="industry"
              placeholder="e.g. SaaS, Finance, Healthcare"
            />

            <button className="btn-block">
              Create organization
            </button>
          </form>
        </div>

        <p className="auth-footer muted">
          Already have an account? <a href="/login">Sign in</a>
        </p>
      </div>
    </div>
  );
}
