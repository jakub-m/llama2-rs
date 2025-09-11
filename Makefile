debug_target=target/debug/llama2-rs
prompt="once upon a time there was a little piggy"

build:
	cargo build --features debug

run: build
	RAYON_NUM_THREADS=1 RUST_BACKTRACE=1 $(debug_target) -z ../llama2.c/tokenizer.bin  ../llama2.c/stories15M.bin  -i '$(prompt)'

debug: build
	lldb $(debug_target) -- -z ../llama2.c/tokenizer.bin  ../llama2.c/stories15M.bin  -i '$(prompt)'
