const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('./jobboard.db');

db.serialize(() => {
  // 1. Create Users Table
  db.run(`CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    role TEXT CHECK(role IN ('employer', 'candidate')) NOT NULL
  )`);

  // 2. Create Jobs Table
  db.run(`CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employer_id INTEGER,
    title TEXT NOT NULL,
    company_name TEXT NOT NULL,
    job_type TEXT,
    location TEXT DEFAULT 'Remote',
    salary_range TEXT,
    tech_stack TEXT,
    description TEXT,
    status TEXT DEFAULT 'active'
  )`);

  // 3. Create Applications Table
  db.run(`CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    candidate_id INTEGER,
    cover_letter TEXT,
    resume_link TEXT,
    status TEXT DEFAULT 'pending',
    UNIQUE(job_id, candidate_id)
  )`);

  // Seed sample initial data if database is empty
  db.get("SELECT COUNT(*) as count FROM users", (err, row) => {
    if (row && row.count === 0) {
      db.run("INSERT INTO users (name, email, role) VALUES ('TechCorp Inc', 'employer@techcorp.com', 'employer')");
      db.run("INSERT INTO users (name, email, role) VALUES ('Jane Doe', 'jane@example.com', 'candidate')");
      
      db.run(`INSERT INTO jobs (employer_id, title, company_name, job_type, location, salary_range, tech_stack, description) 
              VALUES (1, 'Senior React Developer', 'TechCorp Inc', 'Freelance', 'Remote', '$80-$100/hr', 'React, Node.js, SQL', 'Build dynamic user interfaces and integrate backend REST APIs.')`);
      
      db.run(`INSERT INTO jobs (employer_id, title, company_name, job_type, location, salary_range, tech_stack, description) 
              VALUES (1, 'Backend Node.js Engineer', 'TechCorp Inc', 'Full-Time', 'Hybrid', '$120k-$140k', 'Express, SQLite, PostgreSQL', 'Design relational database schemas and optimize query performance.')`);
    }
  });
});

module.exports = db;