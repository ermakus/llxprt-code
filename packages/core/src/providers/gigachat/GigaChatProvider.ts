/**
 * @license
 * Copyright 2025 Vybestack LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * GigaChat Provider for LLxprt
 * Uses the official gigachat package (ai-forever/gigachat) to connect to GigaChat API
 */

import { GigaChat } from 'gigachat';
import type {
  Chat,
  ChatCompletion,
  ChatCompletionChunk,
  ChatFunctionCall,
  ChoicesChunk,
  Choices,
  Message,
  MessageChunk,
  FunctionCall,
  Function as GigaChatFunction,
  FunctionParameters,
  Usage,
  // eslint-disable-next-line import/no-internal-modules -- gigachat package explicitly exports ./interfaces in package.json
} from 'gigachat/interfaces';
// eslint-disable-next-line import/no-internal-modules -- gigachat package exports exceptions
import { GigaChatException } from 'gigachat/exceptions';
import { DebugLogger } from '../../debug/index.js';
import { type IModel } from '../IModel.js';
import {
  type IContent,
  type TextBlock,
  type ToolCallBlock,
  type ToolResponseBlock,
  type ContentBlock,
} from '../../services/history/IContent.js';
import {
  BaseProvider,
  type BaseProviderConfig,
  type NormalizedGenerateChatOptions,
} from '../BaseProvider.js';
import { type IProviderConfig } from '../types/IProviderConfig.js';
import { getCoreSystemPromptAsync } from '../../core/prompts.js';
import { resolveUserMemory } from '../utils/userMemory.js';
import { buildToolResponsePayload } from '../utils/toolResponsePayload.js';
import {
  retryWithBackoff,
  getErrorStatus,
  isNetworkTransientError,
} from '../../utils/retry.js';

export class GigaChatProvider extends BaseProvider {
  private readonly logger = new DebugLogger('llxprt:gigachat');

  constructor(apiKey?: string, baseURL?: string, config?: IProviderConfig) {
    const baseConfig: BaseProviderConfig = {
      name: 'gigachat',
      apiKey,
      baseURL,
      envKeyNames: ['GIGACHAT_API_KEY'],
      isOAuthEnabled: false,
      oauthProvider: undefined,
      oauthManager: undefined,
    };

    super(baseConfig, config);
  }

  /**
   * GigaChat does not support OAuth
   */
  protected supportsOAuth(): boolean {
    return false;
  }

  /**
   * Create a GigaChat client instance
   * The API key (credentials) is the authorization data for OAuth 2.0
   */
  private async createClient(authToken: string): Promise<GigaChat> {
    // GigaChat API uses self-signed certificates, need to allow them
    const https = await import('https');
    const httpsAgent = new https.Agent({
      rejectUnauthorized: false, // Required for GigaChat's self-signed certs
    });

    const client = new GigaChat({
      credentials: authToken,
      scope: 'GIGACHAT_API_PERS', // Personal account scope
      timeout: 60, // 60 seconds timeout
      httpsAgent, // Pass the agent to handle self-signed certs
    });

    // Update the token (performs OAuth 2.0 authentication)
    await client.updateToken();

    return client;
  }

  /**
   * Build provider client per call with fresh SDK instance
   */
  private async buildProviderClient(
    options: NormalizedGenerateChatOptions,
  ): Promise<{ client: GigaChat; authToken: string }> {
    const runtimeAuthToken = options.resolved.authToken;
    let authToken: string | undefined;

    if (
      typeof runtimeAuthToken === 'string' &&
      runtimeAuthToken.trim() !== ''
    ) {
      authToken = runtimeAuthToken;
    } else if (
      runtimeAuthToken &&
      typeof runtimeAuthToken === 'object' &&
      'provide' in runtimeAuthToken &&
      typeof runtimeAuthToken.provide === 'function'
    ) {
      try {
        const freshToken = await runtimeAuthToken.provide();
        if (!freshToken) {
          throw new Error(
            `Auth token unavailable for runtimeId=${options.runtime?.runtimeId}`,
          );
        }
        authToken = freshToken;
      } catch (error) {
        throw new Error(
          `Auth token unavailable for runtimeId=${options.runtime?.runtimeId}: ${error}`,
        );
      }
    }

    if (!authToken) {
      authToken = await this.getAuthToken();
    }

    if (!authToken) {
      throw new Error(
        'No authentication available for GigaChat API calls. Set GIGACHAT_API_KEY environment variable with your Base64-encoded client secret.',
      );
    }

    const client = await this.createClient(authToken);
    return { client, authToken };
  }

  /**
   * No operation - stateless provider has no cache to clear
   */
  clearClientCache(_runtimeKey?: string): void {
    // No-op for stateless provider
  }

  override clearAuthCache(): void {
    super.clearAuthCache();
  }

  override async getModels(): Promise<IModel[]> {
    // GigaChat models - return hardcoded list
    // The API doesn't provide a models listing endpoint
    return [
      {
        id: 'GigaChat-2',
        name: 'GigaChat 2 Lite',
        provider: 'gigachat',
        supportedToolFormats: ['openai'],
        contextWindow: 128000,
        maxOutputTokens: 4096,
      },
      {
        id: 'GigaChat-2-Pro',
        name: 'GigaChat 2 Pro',
        provider: 'gigachat',
        supportedToolFormats: ['openai'],
        contextWindow: 128000,
        maxOutputTokens: 4096,
      },
      {
        id: 'GigaChat-2-Max',
        name: 'GigaChat 2 Max',
        provider: 'gigachat',
        supportedToolFormats: ['openai'],
        contextWindow: 128000,
        maxOutputTokens: 4096,
      },
    ];
  }

  override getCurrentModel(): string {
    return this.getDefaultModel();
  }

  override getDefaultModel(): string {
    return 'GigaChat-2-Max';
  }

  /**
   * GigaChat requires API key (paid mode)
   */
  override isPaidMode(): boolean {
    return true;
  }

  /**
   * Get the list of server tools supported by this provider
   */
  override getServerTools(): string[] {
    return [];
  }

  /**
   * Invoke a server tool (native provider tool)
   */
  override async invokeServerTool(
    _toolName: string,
    _params: unknown,
    _config?: unknown,
    _signal?: AbortSignal,
  ): Promise<unknown> {
    throw new Error('Server tools not supported by GigaChat provider');
  }

  /**
   * Get model parameters from SettingsService per call
   */
  override getModelParams(): Record<string, unknown> | undefined {
    try {
      const settingsService = this.resolveSettingsService();
      const providerSettings = settingsService.getProviderSettings(this.name);

      const reservedKeys = new Set([
        'enabled',
        'apiKey',
        'api-key',
        'apiKeyfile',
        'api-keyfile',
        'baseUrl',
        'base-url',
        'model',
        'toolFormat',
        'tool-format',
        'toolFormatOverride',
        'tool-format-override',
        'defaultModel',
      ]);

      const params: Record<string, unknown> = {};
      if (providerSettings) {
        for (const [key, value] of Object.entries(providerSettings)) {
          if (reservedKeys.has(key) || value === undefined || value === null) {
            continue;
          }
          params[key] = value;
        }
      }

      return Object.keys(params).length > 0 ? params : undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * Check if the provider is authenticated
   */
  override async isAuthenticated(): Promise<boolean> {
    return super.isAuthenticated();
  }

  /**
   * Clean JSON Schema for GigaChat compatibility.
   * GigaChat requires that any object type has a 'properties' field,
   * even if empty. This function recursively ensures all objects have properties.
   */
  private cleanSchemaForGigaChat(schema: unknown): FunctionParameters {
    if (!schema || typeof schema !== 'object') {
      return { type: 'object', properties: {} };
    }

    const input = schema as Record<string, unknown>;
    const result: Record<string, unknown> = {};

    // Copy type (normalize to lowercase)
    const rawType = input.type;
    if (typeof rawType === 'string') {
      result.type = rawType.toLowerCase();
    } else if (typeof rawType === 'number') {
      // Gemini Type enum values
      const typeMap: Record<number, string> = {
        1: 'string',
        2: 'number',
        3: 'integer',
        4: 'boolean',
        5: 'array',
        6: 'object',
      };
      result.type = typeMap[rawType] || 'object';
    } else {
      result.type = 'object';
    }

    // Copy description
    if (typeof input.description === 'string') {
      result.description = input.description;
    }

    // Handle properties recursively - REQUIRED for type: 'object'
    if (result.type === 'object') {
      if (input.properties && typeof input.properties === 'object') {
        const cleanedProps: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(
          input.properties as Record<string, unknown>,
        )) {
          if (value && typeof value === 'object') {
            cleanedProps[key] = this.cleanSchemaForGigaChat(value);
          }
        }
        result.properties = cleanedProps;
      } else {
        // GigaChat requires properties even if empty
        result.properties = {};
      }
    }

    // Handle array items
    if (result.type === 'array' && input.items) {
      if (Array.isArray(input.items)) {
        // Tuple type - use first item
        result.items = this.cleanSchemaForGigaChat(input.items[0]);
      } else {
        result.items = this.cleanSchemaForGigaChat(input.items);
      }
    }

    // Copy required array
    if (Array.isArray(input.required)) {
      result.required = input.required.map((r) => String(r));
    }

    // Copy enum
    if (Array.isArray(input.enum)) {
      result.enum = input.enum;
    }

    // Copy numeric constraints
    if (input.minimum !== undefined) result.minimum = input.minimum;
    if (input.maximum !== undefined) result.maximum = input.maximum;
    if (input.minLength !== undefined) result.minLength = input.minLength;
    if (input.maxLength !== undefined) result.maxLength = input.maxLength;

    // Copy default
    if (input.default !== undefined) {
      result.default = input.default;
    }

    return result as FunctionParameters;
  }

  /**
   * Convert tools to GigaChat function format using official types
   */
  private convertToolsToGigaChat(
    tools?: NormalizedGenerateChatOptions['tools'],
  ): GigaChatFunction[] | undefined {
    if (!tools || tools.length === 0) {
      return undefined;
    }

    const functions: GigaChatFunction[] = [];

    for (const toolGroup of tools) {
      for (const decl of toolGroup.functionDeclarations) {
        const fn: GigaChatFunction = {
          name: decl.name || 'unknown',
          description: decl.description,
        };

        if (decl.parametersJsonSchema) {
          // Clean and convert to FunctionParameters type
          fn.parameters = this.cleanSchemaForGigaChat(
            decl.parametersJsonSchema,
          );
        } else {
          // GigaChat requires parameters, provide empty object schema
          fn.parameters = { type: 'object', properties: {} };
        }

        functions.push(fn);
      }
    }

    return functions.length > 0 ? functions : undefined;
  }

  /**
   * Generate a unique tool call ID
   */
  private generateToolCallId(): string {
    return `call_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Normalize function call arguments from GigaChat.
   * GigaChat sometimes returns escaped slashes in glob patterns (e.g., "**\/*" instead of "** /*").
   * This method recursively unescapes such patterns in string values.
   */
  private normalizeFunctionArguments(
    args: Record<string, unknown>,
  ): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(args)) {
      if (typeof value === 'string') {
        // Unescape backslash-escaped forward slashes: \/ -> /
        result[key] = value.replace(/\\\//g, '/');
      } else if (Array.isArray(value)) {
        result[key] = value.map((item) => {
          if (typeof item === 'string') {
            return item.replace(/\\\//g, '/');
          }
          if (item && typeof item === 'object') {
            return this.normalizeFunctionArguments(
              item as Record<string, unknown>,
            );
          }
          return item;
        });
      } else if (value && typeof value === 'object') {
        result[key] = this.normalizeFunctionArguments(
          value as Record<string, unknown>,
        );
      } else {
        result[key] = value;
      }
    }

    return result;
  }

  /**
   * Get retry configuration
   */
  private getRetryConfig(): { maxAttempts: number; initialDelayMs: number } {
    const ephemeralSettings =
      this.providerConfig?.getEphemeralSettings?.() || {};
    const maxAttempts =
      (ephemeralSettings['retries'] as number | undefined) ?? 6;
    const initialDelayMs =
      (ephemeralSettings['retrywait'] as number | undefined) ?? 4000;
    return { maxAttempts, initialDelayMs };
  }

  /**
   * Extract detailed error message from GigaChat API response errors
   * Uses GigaChatException type from the SDK for proper typing
   *
   * GigaChatException.response.data is http.IncomingMessage (a readable stream)
   * that contains the JSON error body with status and message fields.
   */
  private async extractGigaChatError(error: unknown): Promise<string> {
    // Handle GigaChat SDK exceptions with AxiosResponse
    if (error instanceof GigaChatException) {
      const response = error.response;
      const status = response.status;
      const statusText = response.statusText || '';

      // The data is http.IncomingMessage - a readable stream
      const data = response.data;

      // Try to read the body from the IncomingMessage stream
      if (data && typeof data === 'object' && 'on' in data) {
        try {
          const incomingMessage = data as import('http').IncomingMessage;

          // Read the body from the stream
          const chunks: Buffer[] = [];
          for await (const chunk of incomingMessage) {
            chunks.push(Buffer.from(chunk as Buffer));
          }
          const bodyStr = Buffer.concat(chunks).toString('utf-8');

          if (bodyStr) {
            try {
              const parsed = JSON.parse(bodyStr) as Record<string, unknown>;
              this.logger.debug(() => `[GigaChat Error]`, {
                status,
                statusText,
                body: parsed,
              });

              if (parsed.message)
                return `GigaChat API error (${status}): ${parsed.message}`;
              if (parsed.error)
                return `GigaChat API error (${status}): ${parsed.error}`;
            } catch {
              // Body is not valid JSON, return as-is
              this.logger.debug(
                () => `[GigaChat Error] status=${status}, body=${bodyStr}`,
              );
              return `GigaChat API error (${status}): ${bodyStr}`;
            }
          }
        } catch (readError) {
          this.logger.debug(
            () =>
              `[GigaChat Error] failed to read response stream: ${readError}`,
          );
        }
      }

      return `GigaChat API error: HTTP ${status} ${statusText}`.trim();
    }

    // Handle standard Error
    if (error instanceof Error) {
      return error.message;
    }

    return String(error);
  }

  /**
   * Check if error should trigger retry
   */
  private shouldRetryGigaChatResponse(error: unknown): boolean {
    const status = getErrorStatus(error);
    if (status === 429 || (status && status >= 500 && status < 600)) {
      return true;
    }
    return isNetworkTransientError(error);
  }

  /**
   * Main chat completion implementation
   */
  protected override async *generateChatCompletionWithOptions(
    options: NormalizedGenerateChatOptions,
  ): AsyncIterableIterator<IContent> {
    const { client } = await this.buildProviderClient(options);
    const { contents: content, tools } = options;

    // Convert IContent to GigaChat messages format (using official Message type)
    const messages: Message[] = [];

    const configForMessages =
      options.config ?? options.runtime?.config ?? this.globalConfig;

    // Build messages from content
    for (const c of content) {
      if (c.speaker === 'human') {
        const textBlock = c.blocks.find((b) => b.type === 'text') as
          | TextBlock
          | undefined;
        if (textBlock) {
          messages.push({
            role: 'user',
            content: textBlock.text,
          });
        }
      } else if (c.speaker === 'ai') {
        const textBlocks = c.blocks.filter(
          (b) => b.type === 'text',
        ) as TextBlock[];
        const toolCallBlocks = c.blocks.filter(
          (b) => b.type === 'tool_call',
        ) as ToolCallBlock[];

        if (toolCallBlocks.length > 0) {
          // Handle tool calls - GigaChat supports one function call per message
          // Note: GigaChat requires content to be empty string for function calls
          const tc = toolCallBlocks[0];

          // GigaChat uses object arguments, not JSON strings
          let argsObject: Record<string, unknown>;
          if (typeof tc.parameters === 'string') {
            try {
              argsObject = JSON.parse(tc.parameters) as Record<string, unknown>;
            } catch {
              argsObject = {};
            }
          } else {
            argsObject = (tc.parameters as Record<string, unknown>) || {};
          }

          messages.push({
            role: 'assistant',
            content: '', // GigaChat requires empty string, not null
            function_call: {
              name: tc.name,
              arguments: argsObject,
            },
          });
        } else {
          // Text-only message
          const contentText = textBlocks.map((b) => b.text).join('');
          if (contentText) {
            messages.push({
              role: 'assistant',
              content: contentText,
            });
          }
        }
      } else if (c.speaker === 'tool') {
        const toolResponseBlock = c.blocks.find(
          (b) => b.type === 'tool_response',
        ) as ToolResponseBlock | undefined;

        if (toolResponseBlock) {
          const payload = buildToolResponsePayload(
            toolResponseBlock,
            configForMessages,
          );

          // GigaChat function response format:
          // - role: 'function'
          // - name: function name that was called
          // - content: result string (must be valid JSON)
          //
          // IMPORTANT: GigaChat parses the content as JSON.
          // We wrap the entire payload as a JSON object to ensure valid JSON
          // even if the result text was truncated.
          const functionResponseContent = JSON.stringify({
            status: payload.status,
            result: payload.result || '[empty result]',
            truncated: payload.truncated,
            error: payload.error,
          });

          messages.push({
            role: 'function',
            name: toolResponseBlock.toolName,
            content: functionResponseContent,
          });
        }
      }
    }

    // Get model and system prompt
    const currentModel = options.resolved.model;

    // Get userMemory and system prompt
    const userMemory = await resolveUserMemory(
      options.userMemory,
      () => options.invocation?.userMemory,
    );

    const toolNamesForPrompt =
      tools === undefined
        ? undefined
        : Array.from(
            new Set(
              tools.flatMap((group) =>
                group.functionDeclarations
                  .map((decl) => decl.name)
                  .filter((name): name is string => Boolean(name)),
              ),
            ),
          );

    const systemPrompt = await getCoreSystemPromptAsync(
      userMemory,
      currentModel,
      toolNamesForPrompt,
    );

    // Add system message at the beginning
    if (systemPrompt) {
      messages.unshift({
        role: 'system',
        content: systemPrompt,
      });
    }

    // Ensure we have at least one user message
    if (messages.length === 0 || !messages.some((m) => m.role === 'user')) {
      messages.push({
        role: 'user',
        content: 'Hello',
      });
    }

    // Convert tools to GigaChat format
    const gigaChatFunctions = this.convertToolsToGigaChat(tools);

    // Get streaming setting
    const invocationEphemerals = options.invocation?.ephemerals ?? {};
    const streamingSetting =
      (invocationEphemerals['streaming'] as string | undefined) ??
      this.providerConfig?.getEphemeralSettings?.()?.['streaming'];
    const streamingEnabled = streamingSetting !== 'disabled';

    // Build request body using official Chat type
    const functionCall: ChatFunctionCall | undefined = gigaChatFunctions
      ? 'auto'
      : undefined;
    const requestBody: Chat = {
      model: currentModel,
      messages,
      function_call: functionCall,
      functions: gigaChatFunctions,
    };

    const { maxAttempts, initialDelayMs } = this.getRetryConfig();

    // Log LLM input
    this.logger.debug(
      () =>
        `[GigaChat Request] model=${currentModel}, messages=${messages.length}, functions=${gigaChatFunctions?.length ?? 0}`,
      requestBody,
    );

    if (streamingEnabled) {
      // Streaming mode using official gigachat package's stream() method
      try {
        // The official client.stream() returns an AsyncIterable of ChatCompletionChunk
        const streamApiCall = async () => client.stream(requestBody);

        const stream = await retryWithBackoff(streamApiCall, {
          maxAttempts,
          initialDelayMs,
          shouldRetryOnError: this.shouldRetryGigaChatResponse.bind(this),
          trackThrottleWaitTime: this.throttleTracker,
        });

        let currentFunctionCall: FunctionCall | undefined;
        const collectedChunks: string[] = [];

        // Process stream events - official API yields ChatCompletionChunk directly
        for await (const chunk of stream) {
          const data: ChatCompletionChunk = chunk;
          const choice: ChoicesChunk | undefined = data.choices?.[0];

          if (!choice) {
            continue;
          }

          const delta: MessageChunk = choice.delta;

          // Handle text content
          if (delta.content) {
            const text = delta.content;
            collectedChunks.push(text);
            this.logger.debug(
              () =>
                `[GigaChat Stream] text chunk: ${text.length} chars, content: ${JSON.stringify(text)}`,
            );
            yield {
              speaker: 'ai',
              blocks: [{ type: 'text', text }],
            } as IContent;
          }

          // Handle function calls - using official FunctionCall type
          if (delta.function_call) {
            const fc: FunctionCall = delta.function_call;
            if (fc.name) {
              // Start of a new function call
              currentFunctionCall = {
                name: fc.name,
                arguments: fc.arguments ?? {},
              };
            } else if (fc.arguments && currentFunctionCall) {
              // Additional arguments (merge with existing)
              currentFunctionCall.arguments = {
                ...currentFunctionCall.arguments,
                ...fc.arguments,
              };
            }
          }

          // Emit function call when stream ends with function_call finish reason
          if (choice.finish_reason === 'function_call' && currentFunctionCall) {
            const normalizedArgs = this.normalizeFunctionArguments(
              currentFunctionCall.arguments ?? {},
            );
            this.logger.debug(
              () =>
                `[GigaChat Stream] function_call: ${currentFunctionCall?.name}, args=${JSON.stringify(normalizedArgs)}`,
            );
            yield {
              speaker: 'ai',
              blocks: [
                {
                  type: 'tool_call',
                  id: this.generateToolCallId(),
                  name: currentFunctionCall.name,
                  parameters: normalizedArgs,
                },
              ],
            } as IContent;

            currentFunctionCall = undefined;
          }
        }

        // Log full response after all chunks collected
        if (collectedChunks.length > 0) {
          const fullResponse = collectedChunks.join('');
          this.logger.debug(
            () =>
              `[GigaChat Stream] full response (${collectedChunks.length} chunks, ${fullResponse.length} chars): ${fullResponse}`,
          );
        }
      } catch (error) {
        const errorMessage = await this.extractGigaChatError(error);
        this.logger.debug(() => `[GigaChat Streaming Error] ${errorMessage}`);
        throw new Error(errorMessage);
      }
    } else {
      // Non-streaming mode using official gigachat package's chat() method
      try {
        const apiCall = async (): Promise<ChatCompletion> =>
          client.chat(requestBody);

        const response = await retryWithBackoff(apiCall, {
          maxAttempts,
          initialDelayMs,
          shouldRetryOnError: this.shouldRetryGigaChatResponse.bind(this),
          trackThrottleWaitTime: this.throttleTracker,
        });

        // Log LLM output
        this.logger.debug(() => `[GigaChat Response]`, response);

        const blocks: ContentBlock[] = [];
        const choice: Choices | undefined = response.choices?.[0];

        if (choice?.message) {
          const message: Message = choice.message;

          // Handle text content
          if (message.content) {
            blocks.push({
              type: 'text',
              text: message.content,
            } as TextBlock);
          }

          // Handle function call
          if (message.function_call) {
            const fc: FunctionCall = message.function_call;
            const normalizedArgs = this.normalizeFunctionArguments(
              fc.arguments ?? {},
            );
            blocks.push({
              type: 'tool_call',
              id: this.generateToolCallId(),
              name: fc.name,
              parameters: normalizedArgs,
            } as ToolCallBlock);
          }
        }

        const result: IContent = {
          speaker: 'ai',
          blocks,
        };

        // Add usage metadata using SDK's Usage type
        const usage: Usage | undefined = response.usage;
        if (usage) {
          result.metadata = {
            usage: {
              promptTokens: usage.prompt_tokens,
              completionTokens: usage.completion_tokens,
              totalTokens: usage.total_tokens,
            },
          };
        }

        yield result;
      } catch (error) {
        const errorMessage = await this.extractGigaChatError(error);
        this.logger.debug(() => `[GigaChat Error] ${errorMessage}`);
        throw new Error(errorMessage);
      }
    }
  }
}
