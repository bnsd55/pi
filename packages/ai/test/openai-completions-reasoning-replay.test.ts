import { beforeEach, describe, expect, it, vi } from "vitest";
import { getModel, streamSimple } from "../src/compat.ts";
import type { AssistantMessage, Context, Model, OpenAICompletionsCompat, Usage } from "../src/types.ts";

const mockState = vi.hoisted(() => ({
	lastParams: undefined as unknown,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (params: unknown) => {
					mockState.lastParams = params;
					const stream = {
						async *[Symbol.asyncIterator]() {
							yield {
								choices: [{ delta: {}, finish_reason: "stop" }],
								usage: {
									prompt_tokens: 1,
									completion_tokens: 1,
									prompt_tokens_details: { cached_tokens: 0 },
									completion_tokens_details: { reasoning_tokens: 0 },
								},
							};
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{ data: typeof stream; response: { status: number; headers: Headers } }>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}

	return { default: FakeOpenAI };
});

const emptyUsage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function vllmModel(compat?: OpenAICompletionsCompat): Model<"openai-completions"> {
	return {
		id: "zai-org/glm-5.2",
		name: "GLM 5.2 (local vLLM)",
		api: "openai-completions",
		provider: "local-vllm",
		baseUrl: "http://localhost:8000/v1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 262144,
		maxTokens: 16384,
		compat,
	};
}

/** A turn pi already streamed, with reasoning recorded under the field the response used. */
function contextWithRecordedReasoning(model: Model<"openai-completions">, streamedField: string): Context {
	const assistant: AssistantMessage = {
		role: "assistant",
		content: [
			{ type: "thinking", thinking: "step one", thinkingSignature: streamedField },
			{ type: "text", text: "answer" },
		],
		api: "openai-completions",
		provider: model.provider,
		model: model.id,
		usage: emptyUsage,
		stopReason: "stop",
		timestamp: 2,
	};
	return {
		messages: [
			{ role: "user", content: "hello", timestamp: 1 },
			assistant,
			{ role: "user", content: "continue", timestamp: 3 },
		],
	};
}

async function replayedAssistant(
	model: Model<"openai-completions">,
	streamedField: string,
): Promise<Record<string, unknown>> {
	let payload: unknown;

	await streamSimple(model, contextWithRecordedReasoning(model, streamedField), {
		apiKey: "test",
		onPayload: (params: unknown) => {
			payload = params;
		},
	}).result();

	const params = (payload ?? mockState.lastParams) as { messages: Array<Record<string, unknown>> };
	const assistant = params.messages.find((message) => message.role === "assistant");
	expect(assistant).toBeDefined();
	return assistant as Record<string, unknown>;
}

describe("openai-completions reasoning replay field", () => {
	beforeEach(() => {
		mockState.lastParams = undefined;
	});

	it("echoes reasoning back under the field the response streamed it in", async () => {
		const assistant = await replayedAssistant(vllmModel(), "reasoning");
		expect(assistant.reasoning).toBe("step one");
		expect(assistant.reasoning_content).toBeUndefined();
	});

	it("pins the replay field when reasoningReplayField is set", async () => {
		// vLLM renamed its response field reasoning_content -> reasoning, but chat
		// templates still read reasoning_content. Echoing `reasoning` back renders only
		// while vLLM normalizes the inbound field, and a dropped replay raises no error.
		const assistant = await replayedAssistant(vllmModel({ reasoningReplayField: "reasoning_content" }), "reasoning");
		expect(assistant.reasoning_content).toBe("step one");
		expect(assistant.reasoning).toBeUndefined();
	});

	it("pins reasoning recorded before the field was configured", async () => {
		const assistant = await replayedAssistant(
			vllmModel({ reasoningReplayField: "reasoning_content" }),
			"reasoning_text",
		);
		expect(assistant.reasoning_content).toBe("step one");
		expect(assistant.reasoning_text).toBeUndefined();
	});

	it("keeps replaying opencode-go reasoning as reasoning_content", async () => {
		const model = getModel("opencode-go", "kimi-k2.6");
		expect(model).toBeDefined();
		const assistant = await replayedAssistant(model as Model<"openai-completions">, "reasoning");
		expect(assistant.reasoning_content).toBe("step one");
		expect(assistant.reasoning).toBeUndefined();
	});
});
