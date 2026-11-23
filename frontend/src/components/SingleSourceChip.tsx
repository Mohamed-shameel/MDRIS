/** Grey chip shown when only one modality (image or text) is present. */
export default function SingleSourceChip() {
  return (
    <span
      title="Only one modality (image or text) was available"
      className="inline-flex items-center rounded-full border border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 text-[10px] font-semibold text-slate-500 dark:text-slate-400 whitespace-nowrap"
    >
      Single source
    </span>
  );
}
