import { describe, expect, it, vi } from "vitest";

// Mock the AWS SDK so the test is fully local and deterministic. The mock's
// `send()` returns a stream that yields `messageStart` followed by a
// `messageStop` with a stopReason of our choosing, letting us exercise the
// non-happy paths in streamBedrock without any live AWS credentials.

type StreamEvent = Record<string, unknown>;

const bedrockMock = vi.hoisted(() => ({
	streamEvents: [] as StreamEvent[],
}));

vi.mock("@aws-sdk/client-bedrock-runtime", () => {
	class BedrockRuntimeServiceException extends Error {}

	class BedrockRuntimeClient {
		send(): Promise<{ $metadata: { httpStatusCode: number }; stream: AsyncIterable<StreamEvent> }> {
			const events = [...bedrockMock.streamEvents];
			const stream: AsyncIterable<StreamEvent> = {
				async *[Symbol.asyncIterator]() {
					for (const event of events) {
						yield event;
					}
				},
			};
			return Promise.resolve({
				$metadata: { httpStatusCode: 200 },
				stream,
			});
		}
	}

	class ConverseStreamCommand {
		readonly input: unknown;
		constructor(input: unknown) {
			this.input = input;
		}
	}

	return {
		BedrockRuntimeClient,
		BedrockRuntimeServiceException,
		ConverseStreamCommand,
		StopReason: {
			END_TURN: "end_turn",
			STOP_SEQUENCE: "stop_sequence",
			MAX_TOKENS: "max_tokens",
			MODEL_CONTEXT_WINDOW_EXCEEDED: "model_context_window_exceeded",
			TOOL_USE: "tool_use",
		},
		CachePointType: { DEFAULT: "default" },
		CacheTTL: { ONE_HOUR: "ONE_HOUR" },
		ConversationRole: { ASSISTANT: "assistant", USER: "user" },
		ImageFormat: { JPEG: "jpeg", PNG: "png", GIF: "gif", WEBP: "webp" },
		ToolResultStatus: { ERROR: "error", SUCCESS: "success" },
	};
});

import { getModel } from "../src/models.js";
import { streamBedrock } from "../src/providers/amazon-bedrock.js";
import type { Context } from "../src/types.js";

const context: Context = {
	messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
};

function runWithStopReason(stopReason: string) {
	bedrockMock.streamEvents = [{ messageStart: { role: "assistant" } }, { messageStop: { stopReason } }];
	const model = getModel("amazon-bedrock", "us.anthropic.claude-opus-4-7");
	return streamBedrock(model, context, { cacheRetention: "none" }).result();
}

describe("bedrock stream stopReason error surfacing", () => {
	it("surfaces the raw Bedrock stopReason in the error message for content_filtered", async () => {
		const result = await runWithStopReason("content_filtered");

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("content_filtered");
		// Must not fall back to the generic string that hides the real cause.
		expect(result.errorMessage).not.toContain("An unknown error occurred");
	});

	it("surfaces guardrail_intervened, malformed_model_output, and malformed_tool_use in the error message", async () => {
		for (const reason of ["guardrail_intervened", "malformed_model_output", "malformed_tool_use"]) {
			const result = await runWithStopReason(reason);
			expect(result.stopReason).toBe("error");
			expect(result.errorMessage).toContain(reason);
		}
	});

	it("leaves the happy-path stopReasons untouched", async () => {
		const result = await runWithStopReason("end_turn");
		expect(result.stopReason).toBe("stop");
		expect(result.errorMessage).toBeUndefined();
	});
});
