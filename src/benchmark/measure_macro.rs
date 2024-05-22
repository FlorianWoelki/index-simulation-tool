#[macro_export]
macro_rules! measure_resources {
    ($block:block) => {{
        fn measure(system: &sysinfo::System, pid: usize) {
            print!("\n");
            println!("Measuring resources for process with PID: {}", pid);

            if let Some(process) = system.process(Pid::from(pid)) {
                println!("Current process: {}", process.name());
                let memory_mb = process.memory() as f32 / 1_048_576.0;
                if memory_mb >= 1024f32 {
                    println!("Memory usage: {:.2} GB", memory_mb / 1024.0);
                } else {
                    println!("Memory usage: {:.2} MB", memory_mb);
                }
                println!("CPU usage: {:.2}%", process.cpu_usage());
            }
            print!("\n");
        }

        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        // Sleep for a short period to allow the system to have useful data.
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_all();

        let current_pid = std::process::id() as usize;
        // Measure the resources before running the block.
        measure(&system, current_pid);

        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();

        system.refresh_all();
        measure(&system, current_pid);
        println!("Execution time: {:?}", duration);

        result
    }};
}
