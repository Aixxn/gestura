import { Request } from 'express';

export interface IUser {
  _id?: string;
  email: string;
  username?: string;
  full_name?: string;
  password: string;
  created_at: Date;
}

export interface JwtPayload {
  userId: string;
  email: string;
}

export interface AuthRequest extends Request {
  user?: JwtPayload;
}

export interface LoginInput {
  email: string;
  password: string;
}

export interface RegisterInput {
  email: string;
  password: string;
  full_name?: string;
}
