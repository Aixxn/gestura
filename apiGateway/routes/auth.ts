import express from "express";
import { MongoClient } from "mongodb";

const authRouter = express.Router();

const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017";
const DB_NAME = "gestura";
const USERS_COLLECTION = "users";

let db: any;

const connectMongoDB = async () => {
    try {
        const client = new MongoClient(MONGODB_URI);
        await client.connect();
        db = client.db(DB_NAME);
        console.log("Connected to MongoDB");
    } catch (error) {
        console.error("MongoDB connection error:", error);
    }
};

connectMongoDB();

authRouter.post('/login', async (req, res) => {
    try {
        const { username } = req.body;
        const user = await db.collection(USERS_COLLECTION).findOne({ username });
        
        if (!user) {
            return res.status(401).send({ message: 'Invalid credentials' });
        }
        
        return res.send({ message: 'Login successful', username });
    } catch (error) {
        return res.status(500).send({ message: 'Login error', error });
    }
});

authRouter.post('/register', async (req, res) => {
    try {
        const { username, email, password } = req.body;
        const existingUser = await db.collection(USERS_COLLECTION).findOne({ username });
        
        if (existingUser) {
            return res.status(400).send({ message: 'User already exists' });
        }
        
        await db.collection(USERS_COLLECTION).insertOne({ username, email, password });
        return res.send({ message: 'Registration successful', username, email });
    } catch (error) {
        return res.status(500).send({ message: 'Registration error', error });
    }
});

authRouter.post('/logout', (_req, res) => {
    return res.send({ message: 'Logout successful' });
});

export default authRouter;
