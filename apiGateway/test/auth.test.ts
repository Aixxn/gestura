import 'mocha';
import chai from 'chai';
import chaiHttp from 'chai-http';
import supertest from 'supertest';
import { MongoClient, Db, Collection } from 'mongodb';
import app from '@src/app';

chai.use(chaiHttp);
const { expect } = chai;
const request = supertest(app);

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017';
const DB_NAME = 'gestura';
const USERS_COLLECTION = 'users';

let mongoClient: MongoClient;
let db: Db;
let usersCollection: Collection;

describe('Auth Routes - /api/auth', () => {
  before(async () => {
    // Connect to MongoDB
    mongoClient = new MongoClient(MONGODB_URI);
    await mongoClient.connect();
    db = mongoClient.db(DB_NAME);
    usersCollection = db.collection(USERS_COLLECTION);
    
    // Clear the users collection before tests
    await usersCollection.deleteMany({});
  });

  after(async () => {
    // Clean up and close connection
    await usersCollection.deleteMany({});
    await mongoClient.close();
  });

  beforeEach(async () => {
    // Clear users before each test
    await usersCollection.deleteMany({});
  });

  describe('POST /api/auth/register', () => {
    it('should register a new user successfully', async () => {
      const userData = {
        username: 'testuser',
        email: 'test@example.com',
        password: 'password123'
      };

      const res = await request
        .post('/api/auth/register')
        .send(userData)
        .expect(200);

      expect(res.body).to.have.property('message', 'Registration successful');
      expect(res.body).to.have.property('username', 'testuser');
      expect(res.body).to.have.property('email', 'test@example.com');

      // Verify user was written to MongoDB
      const dbUser = await usersCollection.findOne({ username: 'testuser' });
      expect(dbUser).to.not.be.null;
      expect(dbUser!.username).to.equal('testuser');
      expect(dbUser!.email).to.equal('test@example.com');
      expect(dbUser!.password).to.equal('password123');
    });

    it('should return 400 when username already exists', async () => {
      const userData = {
        username: 'duplicateuser',
        email: 'duplicate@example.com',
        password: 'password123'
      };

      // First registration
      await request
        .post('/api/auth/register')
        .send(userData)
        .expect(200);

      // Second registration with same username
      const res = await request
        .post('/api/auth/register')
        .send(userData)
        .expect(400);

      expect(res.body).to.have.property('message', 'User already exists');
    });

    it('should return 400 for registration with only username', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ username: 'testuser' }) // missing email and password
        .expect(400);

      expect(res.body).to.have.property('message', 'Username, email, and password are required');
    });

    it('should return 400 for registration with username and password only', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ username: 'testuser', password: 'password123' })
        .expect(400);

      expect(res.body).to.have.property('message', 'Username, email, and password are required');
    });

    it('should return 400 for registration with username and email only', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ username: 'testuser', email: 'test@example.com' })
        .expect(400);

      expect(res.body).to.have.property('message', 'Username, email, and password are required');
    });

    it('should store multiple users independently', async () => {
      const user1 = { username: 'user1', email: 'user1@example.com', password: 'pass1' };
      const user2 = { username: 'user2', email: 'user2@example.com', password: 'pass2' };

      await request.post('/api/auth/register').send(user1).expect(200);
      await request.post('/api/auth/register').send(user2).expect(200);

      const count = await usersCollection.countDocuments({});
      expect(count).to.equal(2);

      const dbUser1 = await usersCollection.findOne({ username: 'user1' });
      const dbUser2 = await usersCollection.findOne({ username: 'user2' });

      expect(dbUser1).to.not.be.null;
      expect(dbUser2).to.not.be.null;
      expect(dbUser1!.email).to.equal('user1@example.com');
      expect(dbUser2!.email).to.equal('user2@example.com');
    });
  });

  describe('POST /api/auth/login', () => {
    beforeEach(async () => {
      // Create a test user for login tests
      await usersCollection.insertOne({
        username: 'logintest',
        email: 'login@example.com',
        password: 'correctpassword'
      });
    });

    it('should login successfully with correct credentials', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ username: 'logintest', password: 'correctpassword' })
        .expect(200);

      expect(res.body).to.have.property('message', 'Login successful');
      expect(res.body).to.have.property('username', 'logintest');
    });

    it('should return 401 for non-existent user', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ username: 'nonexistent', password: 'password123' })
        .expect(401);

      expect(res.body).to.have.property('message', 'Invalid credentials');
    });

    it('should return 401 for wrong password', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ username: 'logintest', password: 'wrongpassword' })
        .expect(401);

      expect(res.body).to.have.property('message', 'Invalid credentials');
    });

    it('should return 400 when username is missing', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ password: 'password123' })
        .expect(400);

      expect(res.body).to.have.property('message', 'Username and password are required');
    });

    it('should return 400 when password is missing', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ username: 'logintest' })
        .expect(400);

      expect(res.body).to.have.property('message', 'Username and password are required');
    });

    it('should not modify database on login attempt', async () => {
      const countBefore = await usersCollection.countDocuments({});
      
      await request
        .post('/api/auth/login')
        .send({ username: 'logintest', password: 'correctpassword' })
        .expect(200);

      const countAfter = await usersCollection.countDocuments({});
      expect(countAfter).to.equal(countBefore);
    });
  });

  describe('POST /api/auth/logout', () => {
    it('should return success message on logout', async () => {
      const res = await request
        .post('/api/auth/logout')
        .expect(200);

      expect(res.body).to.have.property('message', 'Logout successful');
    });

    it('should not require authentication', async () => {
      const res = await request
        .post('/api/auth/logout')
        .expect(200);

      expect(res.body).to.have.property('message', 'Logout successful');
    });
  });

  describe('Database Integration', () => {
    it('should verify MongoDB connection is working', async () => {
      const adminDb = mongoClient.db('admin');
      const result = await adminDb.command({ ping: 1 });
      expect(result.ok).to.equal(1);
    });

    it('should verify users collection exists and is accessible', async () => {
      const collections = await db.listCollections({ name: USERS_COLLECTION }).toArray();
      expect(collections.length).to.be.greaterThan(0);
    });

    it('should verify data persistence across requests', async () => {
      // Register a user
      await request
        .post('/api/auth/register')
        .send({ username: 'persistuser', email: 'persist@example.com', password: 'pass123' })
        .expect(200);

      // Verify in database
      const dbUser = await usersCollection.findOne({ username: 'persistuser' });
      expect(dbUser).to.not.be.null;

      // Login with same user
      const loginRes = await request
        .post('/api/auth/login')
        .send({ username: 'persistuser', password: 'pass123' })
        .expect(200);

      expect(loginRes.body.username).to.equal('persistuser');
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed JSON gracefully', async () => {
      await request
        .post('/api/auth/register')
        .set('Content-Type', 'application/json')
        .send('{ invalid json }')
        .expect(400); // Express returns 400 for malformed JSON
    });

    it('should handle empty request body', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({})
        .expect(400);

      expect(res.body).to.have.property('message', 'Username, email, and password are required');
    });
  });
});
