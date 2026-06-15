import 'mocha';
import chai from 'chai';
import supertest from 'supertest';
import { MongoClient, Db, Collection } from 'mongodb';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { MongoMemoryServer } from 'mongodb-memory-server';
import app from '@src/app';

const { expect } = chai;
const request = supertest(app);

const DB_NAME = 'gestura';
const USERS_COLLECTION = 'users';
const JWT_SECRET = process.env.JWT_SECRET || 'default-dev-secret-change-in-production';

let mongoServer: MongoMemoryServer;
let mongoClient: MongoClient;
let db: Db;
let usersCollection: Collection;

describe('Auth Routes - /api/auth', () => {
  before(async function () {
    this.timeout(30000);

    mongoServer = await MongoMemoryServer.create();
    const uri = mongoServer.getUri();
    process.env.MONGODB_URI = uri;

    mongoClient = new MongoClient(uri);
    await mongoClient.connect();
    db = mongoClient.db(DB_NAME);
    usersCollection = db.collection(USERS_COLLECTION);
  });

  after(async () => {
    if (mongoClient) await mongoClient.close();
    if (mongoServer) await mongoServer.stop();
  });

  beforeEach(async () => {
    if (usersCollection) await usersCollection.deleteMany({});
  });

  describe('POST /api/auth/register', () => {
    it('should register a new user and return JWT', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ email: 'test@example.com', password: 'password123', full_name: 'Test User' })
        .expect(201);

      expect(res.body).to.have.property('success', true);
      expect(res.body).to.have.property('message', 'Registration successful');
      expect(res.body).to.have.property('token');
      expect(res.body.user).to.have.property('email', 'test@example.com');
      expect(res.body.user).to.have.property('full_name', 'Test User');
      expect(res.body.user).to.not.have.property('password');

      const decoded = jwt.verify(res.body.token, JWT_SECRET) as any;
      expect(decoded).to.have.property('email', 'test@example.com');
      expect(decoded).to.have.property('userId');

      const dbUser = await usersCollection.findOne({ email: 'test@example.com' });
      expect(dbUser).to.not.be.null;
      expect(dbUser!.password).to.not.equal('password123');
      const isHashed = await bcrypt.compare('password123', dbUser!.password);
      expect(isHashed).to.be.true;
    });

    it('should return 400 when email is missing', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ password: 'password123' })
        .expect(400);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Email and password are required');
    });

    it('should return 400 when password is missing', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ email: 'test@example.com' })
        .expect(400);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Email and password are required');
    });

    it('should return 400 for an empty request body', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({})
        .expect(400);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Email and password are required');
    });

    it('should return 409 when email already exists', async () => {
      await request
        .post('/api/auth/register')
        .send({ email: 'dupe@example.com', password: 'pass123' })
        .expect(201);

      const res = await request
        .post('/api/auth/register')
        .send({ email: 'dupe@example.com', password: 'pass456' })
        .expect(409);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Email already registered');
    });

    it('should normalize email to lowercase', async () => {
      const res = await request
        .post('/api/auth/register')
        .send({ email: 'Test@Example.COM', password: 'pass123' })
        .expect(201);

      expect(res.body.user.email).to.equal('test@example.com');

      const dbUser = await usersCollection.findOne({ email: 'test@example.com' });
      expect(dbUser).to.not.be.null;
    });

    it('should store multiple users independently', async () => {
      await request
        .post('/api/auth/register')
        .send({ email: 'user1@example.com', password: 'pass1' })
        .expect(201);

      await request
        .post('/api/auth/register')
        .send({ email: 'user2@example.com', password: 'pass2' })
        .expect(201);

      const count = await usersCollection.countDocuments({});
      expect(count).to.equal(2);

      const dbUser1 = await usersCollection.findOne({ email: 'user1@example.com' });
      const dbUser2 = await usersCollection.findOne({ email: 'user2@example.com' });

      expect(dbUser1).to.not.be.null;
      expect(dbUser2).to.not.be.null;
    });
  });

  describe('POST /api/auth/login', () => {
    beforeEach(async () => {
      const hashedPassword = await bcrypt.hash('correctpassword', 12);
      await usersCollection.insertOne({
        email: 'login@example.com',
        password: hashedPassword,
        full_name: 'Login User',
        created_at: new Date(),
      });
    });

    it('should login successfully and return JWT', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ email: 'login@example.com', password: 'correctpassword' })
        .expect(200);

      expect(res.body).to.have.property('success', true);
      expect(res.body).to.have.property('message', 'Login successful');
      expect(res.body).to.have.property('token');

      const decoded = jwt.verify(res.body.token, JWT_SECRET) as any;
      expect(decoded).to.have.property('email', 'login@example.com');
      expect(decoded).to.have.property('userId');

      expect(res.body.user).to.not.have.property('password');
    });

    it('should return 401 for wrong password', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ email: 'login@example.com', password: 'wrongpassword' })
        .expect(401);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Invalid email or password');
    });

    it('should return 401 for non-existent email', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ email: 'nonexistent@example.com', password: 'password123' })
        .expect(401);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Invalid email or password');
    });

    it('should return 400 when email is missing', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ password: 'password123' })
        .expect(400);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Email and password are required');
    });

    it('should return 400 when password is missing', async () => {
      const res = await request
        .post('/api/auth/login')
        .send({ email: 'login@example.com' })
        .expect(400);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Email and password are required');
    });

    it('should not modify database on login attempt', async () => {
      const countBefore = await usersCollection.countDocuments({});

      await request
        .post('/api/auth/login')
        .send({ email: 'login@example.com', password: 'correctpassword' })
        .expect(200);

      const countAfter = await usersCollection.countDocuments({});
      expect(countAfter).to.equal(countBefore);
    });
  });

  describe('POST /api/auth/logout', () => {
    it('should return success message', async () => {
      const res = await request.post('/api/auth/logout').expect(200);

      expect(res.body).to.have.property('success', true);
      expect(res.body).to.have.property('message', 'Logout successful');
    });
  });

  describe('GET /api/auth/me', () => {
    it('should return user profile with valid token', async () => {
      const registerRes = await request
        .post('/api/auth/register')
        .send({ email: 'me@example.com', password: 'pass123', full_name: 'Me User' })
        .expect(201);

      const token = registerRes.body.token;

      const res = await request
        .get('/api/auth/me')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      expect(res.body).to.have.property('success', true);
      expect(res.body.user).to.have.property('email', 'me@example.com');
      expect(res.body.user).to.not.have.property('password');
    });

    it('should return 401 without authorization header', async () => {
      const res = await request.get('/api/auth/me').expect(401);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'No authorization header provided');
    });

    it('should return 401 with invalid token', async () => {
      const res = await request
        .get('/api/auth/me')
        .set('Authorization', 'Bearer invalid-token')
        .expect(401);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Invalid token');
    });

    it('should return 401 with malformed auth header', async () => {
      const res = await request
        .get('/api/auth/me')
        .set('Authorization', 'InvalidFormat token')
        .expect(401);

      expect(res.body).to.have.property('success', false);
      expect(res.body).to.have.property('message', 'Invalid authorization format. Use: Bearer <token>');
    });
  });
});
