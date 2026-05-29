import {
  addDoc,
  collection,
  getDocs,
  limit,
  query,
  where,
} from "firebase/firestore";

import { db } from "./firebase";

export interface RegisterInput {
  email: string;
  full_name: string;
  password: string;
}

export interface LoginInput {
  email: string;
  password: string;
}

export async function registerUser(input: RegisterInput) {
  const email = input.email.trim().toLowerCase();
  const full_name = input.full_name.trim();
  const password = input.password;

  if (!email || !full_name || !password) {
    throw new Error('Please fill in all required fields.');
  }

  const usersRef = collection(db, "users");
  const existingQuery = query(usersRef, where("email", "==", email), limit(1));
  const existingSnapshot = await getDocs(existingQuery);
  if (!existingSnapshot.empty) {
    throw new Error("Email already registered.");
  }

  const newUser = {
    email,
    full_name,
    password,
    created_at: new Date(),
  };

  const docRef = await addDoc(usersRef, newUser);

  return { _id: docRef.id, email, full_name };
}

export async function loginUser(input: LoginInput) {
  const email = input.email.trim().toLowerCase();
  const password = input.password;

  if (!email || !password) {
    throw new Error('Please enter your email and password.');
  }

  const usersRef = collection(db, "users");
  const loginQuery = query(
    usersRef,
    where("email", "==", email),
    where("password", "==", password),
    limit(1),
  );
  const snapshot = await getDocs(loginQuery);
  if (snapshot.empty) {
    throw new Error("Invalid email or password.");
  }

  const doc = snapshot.docs[0];
  const user = doc?.data();
  return { _id: doc?.id, ...user };
}
