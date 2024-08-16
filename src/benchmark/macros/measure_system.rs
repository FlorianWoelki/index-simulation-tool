use std::time::Duration;

#[derive(Debug)]
pub struct ResourceReport {
    pub initial_memory: f32,
    pub final_memory: f32,
    pub initial_cpu: f32,
    pub final_cpu: f32,
    pub execution_time: Duration,
}

#[macro_export]
macro_rules! measure_resources {
    ($block:block) => {{
        use benchmark::macros::measure_system::ResourceReport;

        fn measure(system: &sysinfo::System, pid: usize) -> (f32, f32) {
            if let Some(process) = system.process(Pid::from(pid)) {
                let memory_mb = process.memory() as f32 / (1024.0 * 1024.0);
                let num_threads = num_cpus::get();
                let cpu_usage = process.cpu_usage() as f32 / num_threads as f32;
                return (memory_mb, cpu_usage);
            }
            (0.0, 0.0)
        }

        let mut system = sysinfo::System::new_all();
        system.refresh_all();
        // Sleep for a short period to allow the system to have useful data.
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        system.refresh_all();

        let current_pid = std::process::id() as usize;
        // Measure the resources before running the block.
        let (initial_memory, initial_cpu) = measure(&system, current_pid);

        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();

        system.refresh_all();
        let (final_memory, final_cpu) = measure(&system, current_pid);

        let report = ResourceReport {
            initial_memory,
            final_memory,
            initial_cpu,
            final_cpu,
            execution_time: duration,
        };

        (result, report)
    }};
}
