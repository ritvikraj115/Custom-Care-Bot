import express from "express";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import Client from "../models/Client.js";

const router = express.Router();

// register
router.post("/register", async (req, res) => {
  const { companyName, email, password, industry } = req.body;

  const exists = await Client.findOne({ email });
  if (exists) return res.status(400).json({ message: "Email already used" });

  const passwordHash = await bcrypt.hash(password, 10);
  const client = await Client.create({
    companyName,
    email,
    passwordHash,
    industry
  });

  res.json({ message: "Registered successfully" });
});

// login
router.post("/login", async (req, res) => {
  const { email, password } = req.body;

  const client = await Client.findOne({ email });
  if (!client) return res.status(400).json({ message: "Invalid credentials" });

  const match = await bcrypt.compare(password, client.passwordHash);
  if (!match) return res.status(400).json({ message: "Invalid credentials" });

  const token = jwt.sign(
    { clientId: client._id },
    process.env.JWT_SECRET,
    { expiresIn: "1d" }
  );

  res.json({ token });
});

export default router;

