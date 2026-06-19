import express from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { MongoClient, ObjectId } from 'mongodb';
import { IUser, AuthRequest, RegisterInput } from 'types/index';
import { requireAuth } from 'middleware/auth';
import logger from 'services/logger';
import { withLogging } from 'middleware/logging';

const authRouter = express.Router();

const DB_NAME = 'gestura';
const USERS_COLLECTION = 'users';
const JWT_SECRET = process.env.JWT_SECRET || 'default-dev-secret-change-in-production';
const JWT_EXPIRES_IN = 7 * 24 * 60 * 60; // 7 days in seconds
const SALT_ROUNDS = 12;

let client: MongoClient | null = null;
let db: any = null;

async function getDb(): Promise<any> {
  if (db) return db;

  const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017';
  client = new MongoClient(uri);
  await client.connect();
  db = client.db(DB_NAME);
  logger.info('Connected to MongoDB');
  return db;
}

function signToken(user: IUser): string {
  return jwt.sign(
    { userId: String(user._id), email: user.email },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
}

function sanitizeUser(user: any) {
  const { password, ...safe } = user;
  return {
    ...safe,
    _id: String(safe._id),
  };
}

authRouter.post('/register', withLogging(async (req, res) => {
  try {
    const database = await getDb();
    const { email, password, full_name } = req.body as RegisterInput;

    if (!email || !password) {
      res.status(400).json({ success: false, message: 'Email and password are required' });
      return;
    }

    const normalizedEmail = email.trim().toLowerCase();

    const existingUser = await database.collection(USERS_COLLECTION).findOne({ email: normalizedEmail });
    if (existingUser) {
      res.status(409).json({ success: false, message: 'Email already registered' });
      return;
    }

    const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
    const createdAt = new Date();

    const result = await database.collection(USERS_COLLECTION).insertOne({
      email: normalizedEmail,
      password: hashedPassword,
      full_name: full_name || '',
      created_at: createdAt,
    });

    const newUser: IUser = {
      _id: result.insertedId.toString(),
      email: normalizedEmail,
      full_name: full_name || '',
      password: hashedPassword,
      created_at: createdAt,
    };

    const token = signToken(newUser);

    res.status(201).json({
      success: true,
      message: 'Registration successful',
      token,
      user: sanitizeUser(newUser),
    });
  } catch (error) {
    logger.error('Registration error', { error });
    res.status(500).json({ success: false, message: 'Registration error' });
  }
}, 'auth.register'));

authRouter.post('/login', withLogging(async (req, res) => {
  try {
    const database = await getDb();
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ success: false, message: 'Email and password are required' });
      return;
    }

    const normalizedEmail = email.trim().toLowerCase();

    const user = await database.collection(USERS_COLLECTION).findOne({ email: normalizedEmail });
    if (!user) {
      res.status(401).json({ success: false, message: 'Invalid email or password' });
      return;
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      res.status(401).json({ success: false, message: 'Invalid email or password' });
      return;
    }

    const token = signToken(user);

    res.json({
      success: true,
      message: 'Login successful',
      token,
      user: sanitizeUser(user),
    });
  } catch (error) {
    logger.error('Login error', { error });
    res.status(500).json({ success: false, message: 'Login error' });
  }
}, 'auth.login'));

authRouter.post('/logout', withLogging((_req, res) => {
  res.json({ success: true, message: 'Logout successful' });
}, 'auth.logout'));

authRouter.get('/me', requireAuth, withLogging(async (req: AuthRequest, res) => {
  try {
    const database = await getDb();
    const user = await database.collection(USERS_COLLECTION).findOne(
      { _id: new ObjectId(req.user!.userId) },
      { projection: { password: 0 } }
    );

    if (!user) {
      res.status(404).json({ success: false, message: 'User not found' });
      return;
    }

    res.json({ success: true, user: sanitizeUser(user) });
  } catch (error) {
    logger.error('Get profile error', { error });
    res.status(500).json({ success: false, message: 'Failed to get user profile' });
  }
}, 'auth.me'));

export default authRouter;
