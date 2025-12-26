#!/usr/bin/env node
/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Entry point for the A2A server binary.
 * This file is referenced in package.json's bin field.
 */

import { logger } from './utils/logger.js';
import { main } from './http/app.js';

process.on('uncaughtException', (error) => {
  logger.error('Unhandled exception:', error);
  process.exit(1);
});

main().catch((error) => {
  logger.error('[CoreAgent] Unhandled error in main:', error);
  process.exit(1);
});
